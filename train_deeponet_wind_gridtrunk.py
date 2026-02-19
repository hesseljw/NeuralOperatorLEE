"""
Train a DeepONet-style surrogate for wind-only outdoor acoustics (LEE data).

Model: branch network encodes the wind profile U(z); trunk is a grid-based spectral (FNO-like) network over (x,z),
merged via a dot-product to predict complex pressure on the ROI grid.

Targets:
- direct complex pressure p(x,z)  OR
- residual Δp(x,z)=p_wind(x,z)-p_no-wind(x,z) (reconstructed as p = p0 + Δp)

Outputs are saved under runs/<RUN_TAG>/ (checkpoints, metrics, eval summaries).

Quickstart:
  python train_deeponet_wind_gridtrunk.py --data-root <DATA_DIR> --iid --device cuda
"""

from __future__ import annotations

import json
import time
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# USER CONFIG
# =============================================================================
# Edit the values below for your machine / experiment. You can also override many
# of them via command-line flags (see --help).

DATA_ROOT = Path("data") / "wind_sobol_complex_n_1000"
DEVICE    = None  # "cuda" or "cpu" (if None auto)
RUN_TAG   = "deeponet_wind_gridtrunk_fno_learn_p_iid"

RUN_IID      = True
RUN_LOFO_ALL = False

# ------------------------------ Resume toggles ------------------------------
RESUME_TRAINING  = True
RESUME_FROM_BEST = False  # if True resume from best instead of last


CFG: Dict[str, Any] = {
    "seed": 1234,

    # ---- Training ----
    "epochs": 3000,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_cases": 1,
    "num_workers": 0,

    # ---- Early stopping (optional) ----
    "early_stop": False,
    "patience": 40,
    "es_delta": 1e-5,

    # ---- AMP ----
    "amp": True,  # autocast ON, GradScaler auto-disabled if complex params exist

    # ---- LR scheduler (ReduceLROnPlateau) ----
    "use_lr_scheduler": True,
    "sched_factor": 0.75,
    "sched_patience": 12,
    "sched_threshold": 1e-5,
    "sched_min_lr": 1e-6,
    "sched_cooldown": 0,

    # ---- Wind sensors (branch) ----
    # number of sensor points along z where U(z) is sampled
    "branch_m": 128,

    # include shear features (recommended): concatenate dU/dz sensors to U sensors
    "use_dudz": True,

    # ---- Wind normalization (matches wind-FNO) ----
    # mode: "scale" uses U_norm=U/u_scale
    #       "zscore" uses U_norm=(U-mu)/std over train set (mu/std stored in stats)
    "normalize_u": True,
    "u_norm_mode": "scale",
    "u_scale": 25.0,

    # ---- Masking flags (kept for parity; wind ROI is typically all-air) ----
    "masked_loss": False,
    "masked_stats": False,
    "masked_metrics": False,

    # ---- p vs delta ----
    # If True: learn delta to baseline p0 (flat ground, no wind): Δp = p - p0
    # If False: learn p directly.
    "predict_delta": False,
    "baseline_file": "deeponet_aux/flat_ground.h5",  # relative to DATA_ROOT

    # ---- Trunk input features ----
    # Strong analogue to terrain DeepONet: provide the baseline p0 (normalized) as trunk channels.
    "use_p0_trunk": True,
    "p0_feat_norm": "zscore",  # currently only "zscore" is used

    # ---- Optional crop to safe window (matches wind-FNO feature) ----
    "crop_to_safe_x_window": False,
    "safe_x_window": (-450.0, 450.0),

    # ---- DeepONet latent ----
    "latent": 64,

    # ---- Branch MLP ----
    "branch_width": 256,
    "branch_depth": 4,

    # ---- Grid Fourier trunk (FNO-style) ----
    "trunk_width": 64,
    "trunk_layers": 4,
    "modes_z": 24,
    "modes_x": 64,
    "padding_frac": 0.05,

    # ---- Debug ----
    "debug_overfit": False,
    "debug_n_cases": 8,
    "debug_constant_target": False,
    "debug_constant_re": 0.0,
    "debug_constant_im": 0.0,
    "debug_shapes": False,
}


# ------------------------------ Utils / I/O ------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def device_info_string(device: str) -> str:
    try:
        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            cc = f"{props.major}.{props.minor}"
            mem_gb = props.total_memory // (1024**3)
            return f"cuda:{idx} ({props.name}, cc={cc}, {mem_gb}GB)"
        return "cpu"
    except Exception:
        return str(device)


def find_cases(data_root: Path) -> List[Path]:
    return sorted((data_root / "cases").glob("case_*__*.h5"))


def find_data_root() -> Path:
    base = Path("data")
    if not base.exists():
        raise FileNotFoundError("./data not found; set DATA_ROOT to your wind run.")

    candidates: List[Path] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        cases = list((d / "cases").glob("case_*__*.h5"))
        if not cases:
            continue
        try:
            with h5py.File(cases[0], "r") as h:
                if "u_profile" in h and "p_re" in h and "p_im" in h and "x_grid" in h and "z_grid" in h:
                    candidates.append(d)
        except Exception:
            pass

    if not candidates:
        raise FileNotFoundError(
            "No complex wind dataset found under ./data/<RUN_NAME>/cases (expected u_profile + p_re + p_im)."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    print(f"[autorun] Using data root: {chosen}")
    return chosen


def case_family(h5_path: Path) -> str:
    name = h5_path.name
    if "__" in name:
        suf = name.split("__", 1)[-1]
        fam = suf.split(".")[0]
        if fam:
            return fam
    try:
        with h5py.File(h5_path, "r") as h:
            if "meta" in h:
                attrs = h["meta"].attrs
                if "family" in attrs:
                    v = attrs["family"]
                    if isinstance(v, (bytes, bytearray)):
                        return v.decode("utf-8")
                    return str(v)
    except Exception:
        pass
    return "u?"


def load_manifest(data_root: Path) -> Dict[str, Any]:
    man = data_root / "manifest.json"
    if man.exists():
        return json.loads(man.read_text())
    return {}


def write_list(p: Path, items: List[Path]):
    ensure_dir(p.parent)
    p.write_text("\n".join(str(x) for x in items))


def read_list(p: Path) -> List[Path]:
    return [Path(s) for s in p.read_text().strip().splitlines() if s.strip()]


def load_grids(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h:
        xg = np.array(h["x_grid"][:], dtype=np.float32)
        zg = np.array(h["z_grid"][:], dtype=np.float32)
    return xg, zg


def load_u_profile(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h:
        zz = np.array(h["u_profile"]["z"][:], dtype=np.float32)
        uu = np.array(h["u_profile"]["U"][:], dtype=np.float32)
    return zz, uu


def load_p_re_im(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h:
        pre = np.array(h["p_re"][:], dtype=np.float32)
        pim = np.array(h["p_im"][:], dtype=np.float32)
    return pre, pim


def ensure_field_zx(field: np.ndarray, xg: np.ndarray, zg: np.ndarray) -> np.ndarray:
    H, W = len(zg), len(xg)
    if field.shape == (H, W):
        return field
    if field.shape == (W, H):
        return field.T
    raise ValueError(f"Field shape {field.shape} incompatible with (H,W)=({H},{W})")


def ensure_complex_zx(pre: np.ndarray, pim: np.ndarray, xg: np.ndarray, zg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return ensure_field_zx(pre, xg, zg), ensure_field_zx(pim, xg, zg)


def x_crop_indices(xg: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return np.where((xg >= xmin) & (xg <= xmax))[0]


def _assert_same_grid(xg_a: np.ndarray, zg_a: np.ndarray, xg_b: np.ndarray, zg_b: np.ndarray, name_a: str, name_b: str):
    if xg_a.shape != xg_b.shape or zg_a.shape != zg_b.shape:
        raise ValueError(f"Grid shape mismatch {name_a} vs {name_b}.")
    if not np.allclose(xg_a, xg_b) or not np.allclose(zg_a, zg_b):
        raise ValueError(f"Grid values mismatch {name_a} vs {name_b}.")


# ------------------------------ Splits ------------------------------

def make_splits(data_root: Path, cfg=CFG):
    cases = find_cases(data_root)
    assert cases, f"No cases found under {data_root}/cases"

    by_family: Dict[str, List[Path]] = {}
    for p in cases:
        by_family.setdefault(case_family(p), []).append(p)

    rng = np.random.default_rng(int(cfg["seed"]))

    iid_dir = data_root / "splits" / "iid"
    if not (iid_dir / "train.txt").exists():
        train, val, test = [], [], []
        for fam, lst in by_family.items():
            lst = list(lst)
            rng.shuffle(lst)
            n = len(lst)
            ntr = max(1, int(0.70 * n))
            nva = max(1, int(0.15 * n))
            train += lst[:ntr]
            val   += lst[ntr:ntr + nva]
            test  += lst[ntr + nva:]
        write_list(iid_dir / "train.txt", train)
        write_list(iid_dir / "val.txt", val)
        write_list(iid_dir / "test.txt", test)
        print(f"[splits] Wrote IID: {len(train)}/{len(val)}/{len(test)}")
    else:
        print("[splits] IID already exists -> skip")

    lofo_dir = data_root / "splits" / "lofo"
    for fam in sorted(by_family):
        fam_dir = lofo_dir / fam
        if (fam_dir / "train.txt").exists():
            continue
        test = list(by_family[fam])
        train_pool = [p for f, lst in by_family.items() if f != fam for p in lst]
        rng.shuffle(train_pool)
        nva = max(1, int(0.15 * len(train_pool)))
        val = train_pool[:nva]
        train = train_pool[nva:]
        write_list(fam_dir / "train.txt", train)
        write_list(fam_dir / "val.txt", val)
        write_list(fam_dir / "test.txt", test)
        print(f"[splits] Wrote LOFO/{fam}: {len(train)}/{len(val)}/{len(test)}")


# ------------------------------ Baseline p0 (flat + no wind) ------------------------------

def _baseline_path(data_root: Path, cfg: Dict[str, Any]) -> Path:
    rel = str(cfg.get("baseline_file", "deeponet_aux/flat_ground.h5"))
    p = Path(rel)
    return p if p.is_absolute() else (data_root / p)


def load_baseline_p0(data_root: Path, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p0_path = _baseline_path(data_root, cfg)
    if not p0_path.exists():
        raise FileNotFoundError(f"Baseline p0 not found: {p0_path}")
    xg, zg = load_grids(p0_path)
    pre_raw, pim_raw = load_p_re_im(p0_path)
    pre, pim = ensure_complex_zx(pre_raw, pim_raw, xg, zg)
    return xg, zg, pre, pim


# ------------------------------ Streaming stats ------------------------------

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).ravel()
        n2 = int(x.size)
        if n2 == 0:
            return
        mean2 = float(x.mean())
        var2  = float(x.var(ddof=0))
        M2_2  = var2 * n2

        if self.n == 0:
            self.n = n2
            self.mean = mean2
            self.M2 = M2_2
            return

        n1 = self.n
        mean1 = self.mean
        M2_1 = self.M2

        n = n1 + n2
        delta = mean2 - mean1
        mean = mean1 + delta * (n2 / n)
        M2 = M2_1 + M2_2 + delta * delta * (n1 * n2 / n)

        self.n = n
        self.mean = mean
        self.M2 = M2

    def finalize(self) -> Tuple[float, float]:
        if self.n < 2:
            return float(self.mean), 1.0
        var = self.M2 / self.n
        std = math.sqrt(max(var, 1e-12))
        return float(self.mean), float(std)


def compute_stats_from_train(
    train_list: List[Path],
    sensors_z: np.ndarray,
    cfg: Dict[str, Any],
    *,
    data_root: Path,
    p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    masked_stats = bool(cfg.get("masked_stats", False))
    predict_delta = bool(cfg.get("predict_delta", False))
    use_p0_trunk = bool(cfg.get("use_p0_trunk", True))

    if (predict_delta or use_p0_trunk) and p0 is None:
        p0 = load_baseline_p0(data_root, cfg)

    rs_re = RunningStats()
    rs_im = RunningStats()

    # wind zscore stats (only needed when requested)
    u_vals: List[np.ndarray] = []

    for path in train_list:
        xg, zg = load_grids(path)
        pre_raw, pim_raw = load_p_re_im(path)
        pre, pim = ensure_complex_zx(pre_raw, pim_raw, xg, zg)

        idx_x = None
        if cfg.get("crop_to_safe_x_window", False):
            xmin, xmax = cfg.get("safe_x_window", (-450.0, 450.0))
            idx_x = x_crop_indices(xg, float(xmin), float(xmax))
            if idx_x.size >= 16:
                xg = xg[idx_x]
                pre = pre[:, idx_x]
                pim = pim[:, idx_x]

        if predict_delta:
            assert p0 is not None
            xg0, zg0, p0_re, p0_im = p0
            if idx_x is not None and idx_x.size >= 16:
                xg0 = xg0[idx_x]
                p0_re = p0_re[:, idx_x]
                p0_im = p0_im[:, idx_x]
            _assert_same_grid(xg, zg, xg0, zg0, str(path), "baseline_p0")
            pre = pre - p0_re
            pim = pim - p0_im

        if masked_stats:
            # wind-only ROI is usually all-air, but we keep this hook
            mask = np.ones_like(pre, dtype=bool)
            rs_re.update(pre[mask])
            rs_im.update(pim[mask])
        else:
            rs_re.update(pre.reshape(-1))
            rs_im.update(pim.reshape(-1))

        if bool(cfg.get("normalize_u", True)) and str(cfg.get("u_norm_mode", "scale")).lower() == "zscore":
            z_u, U = load_u_profile(path)
            # bring U onto ROI z_grid for consistency
            if len(U) != len(zg):
                U = np.interp(zg.astype(np.float64), z_u.astype(np.float64), U.astype(np.float64)).astype(np.float32)
            u_vals.append(U.reshape(-1).astype(np.float32))

    p_re_mean, p_re_std = rs_re.finalize()
    p_im_mean, p_im_std = rs_im.finalize()

    u_mu, u_std = 0.0, 1.0
    if u_vals:
        u_cat = np.concatenate(u_vals)
        u_mu = float(u_cat.mean())
        u_std = float(u_cat.std() + 1e-8)

    stats: Dict[str, Any] = {
        "p_re_mean": p_re_mean,
        "p_re_std":  p_re_std,
        "p_im_mean": p_im_mean,
        "p_im_std":  p_im_std,

        "predict_delta": bool(predict_delta),
        "baseline_file": str(cfg.get("baseline_file", "")) if predict_delta else "",

        "use_dudz": bool(cfg.get("use_dudz", True)),
        "normalize_u": bool(cfg.get("normalize_u", True)),
        "u_norm_mode": str(cfg.get("u_norm_mode", "scale")),
        "u_scale": float(cfg.get("u_scale", 25.0)),
        "u_mu": u_mu,
        "u_std": u_std,

        "masked_stats": bool(masked_stats),
        "crop_to_safe_x_window": bool(cfg.get("crop_to_safe_x_window", False)),
        "safe_x_window": tuple(cfg.get("safe_x_window", (-450.0, 450.0))),

        "branch_m": int(cfg.get("branch_m", len(sensors_z))),
    }

    if use_p0_trunk:
        assert p0 is not None
        _, _, p0_re, p0_im = p0
        stats.update({
            "use_p0_trunk": True,
            "p0_feat_norm": str(cfg.get("p0_feat_norm", "zscore")),
            "p0_re_mean": float(p0_re.mean()),
            "p0_re_std":  float(p0_re.std() + 1e-8),
            "p0_im_mean": float(p0_im.mean()),
            "p0_im_std":  float(p0_im.std() + 1e-8),
        })

    return stats


# ------------------------------ Sensors (z) ------------------------------

def uniform_sensors_z(zmin: float, zmax: float, m: int) -> np.ndarray:
    return np.linspace(zmin, zmax, int(m), dtype=np.float32)


# ------------------------------ Dataset ------------------------------

class WindDeepONetGridDataset(Dataset):
    """Per-case returns:
      branch_input: (m) or (2m) (U_sensors and optional dU/dz_sensors) in normalized space
      y:            (2,H,W) normalized target (p or ΔΔp)
      mask:         (H,W) (all ones by default)
    """

    def __init__(
        self,
        paths: List[Path],
        sensors_z: np.ndarray,
        stats: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        data_root: Path,
        p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        self.paths = list(paths)
        self.sensors_z = sensors_z.astype(np.float32)
        self.stats = stats
        self.cfg = cfg
        self.data_root = data_root

        self.predict_delta = bool(cfg.get("predict_delta", False))
        self.use_dudz = bool(cfg.get("use_dudz", True))
        self.normalize_u = bool(cfg.get("normalize_u", True))
        self.u_norm_mode = str(cfg.get("u_norm_mode", "scale")).lower()
        self.u_scale = float(cfg.get("u_scale", 25.0))

        self.p0 = p0
        if (self.predict_delta or bool(cfg.get("use_p0_trunk", True))) and self.p0 is None:
            self.p0 = load_baseline_p0(data_root, cfg)

        # Precompute branch inputs (like terrain DeepONet does)
        bis: List[np.ndarray] = []
        for path in self.paths:
            xg, zg = load_grids(path)
            z_u, U = load_u_profile(path)

            # bring U onto ROI z_grid (robust)
            if len(U) != len(zg):
                U = np.interp(zg.astype(np.float64), z_u.astype(np.float64), U.astype(np.float64)).astype(np.float32)

            z_range = float(zg.max() - zg.min() + 1e-8)

            # sample U on sensors_z using the ROI z-grid
            U_s = np.interp(self.sensors_z.astype(np.float64), zg.astype(np.float64), U.astype(np.float64)).astype(np.float32)

            # normalize U_s
            if not self.normalize_u:
                U_s_norm = U_s
            elif self.u_norm_mode == "zscore":
                mu = float(self.stats.get("u_mu", 0.0))
                sd = float(self.stats.get("u_std", 1.0))
                U_s_norm = (U_s - mu) / (sd + 1e-12)
            else:
                U_s_norm = U_s / (self.u_scale + 1e-12)

            parts = [U_s_norm.astype(np.float32)]

            if self.use_dudz:
                dudz = np.gradient(U.astype(np.float64), zg.astype(np.float64)).astype(np.float32)
                dudz_s = np.interp(self.sensors_z.astype(np.float64), zg.astype(np.float64), dudz.astype(np.float64)).astype(np.float32)

                if not self.normalize_u:
                    dudz_s_norm = dudz_s
                elif self.u_norm_mode == "zscore":
                    sd = float(self.stats.get("u_std", 1.0))
                    # scale derivative by z_range to keep it comparable in magnitude (same idea as wind-FNO)
                    dudz_s_norm = dudz_s * (z_range / (sd + 1e-12))
                else:
                    dudz_s_norm = dudz_s * (z_range / (self.u_scale + 1e-12))

                parts.append(dudz_s_norm.astype(np.float32))

            bi = np.concatenate(parts, axis=0).astype(np.float32)  # (m) or (2m)
            bis.append(bi)

        self.branch_inputs = np.stack(bis, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        xg, zg = load_grids(path)
        pre_raw, pim_raw = load_p_re_im(path)
        pre, pim = ensure_complex_zx(pre_raw, pim_raw, xg, zg)

        idx_x = None
        if self.cfg.get("crop_to_safe_x_window", False):
            xmin, xmax = self.cfg.get("safe_x_window", (-450.0, 450.0))
            idx_x = x_crop_indices(xg, float(xmin), float(xmax))
            if idx_x.size >= 16:
                xg = xg[idx_x]
                pre = pre[:, idx_x]
                pim = pim[:, idx_x]

        if self.predict_delta:
            assert self.p0 is not None
            xg0, zg0, p0_re, p0_im = self.p0
            if idx_x is not None and idx_x.size >= 16:
                xg0 = xg0[idx_x]
                p0_re = p0_re[:, idx_x]
                p0_im = p0_im[:, idx_x]
            _assert_same_grid(xg, zg, xg0, zg0, str(path), "baseline_p0")
            pre = pre - p0_re
            pim = pim - p0_im

        H, W = pre.shape

        # normalize target
        y_re = (pre - float(self.stats["p_re_mean"])) / float(self.stats["p_re_std"])
        y_im = (pim - float(self.stats["p_im_mean"])) / float(self.stats["p_im_std"])

        if bool(self.cfg.get("debug_constant_target", False)):
            y_re[:] = float(self.cfg.get("debug_constant_re", 0.0))
            y_im[:] = float(self.cfg.get("debug_constant_im", 0.0))

        y = np.stack([y_re.astype(np.float32), y_im.astype(np.float32)], axis=0)  # (2,H,W)

        mask = np.ones((H, W), dtype=np.float32)
        b = self.branch_inputs[idx]
        return {"branch_input": b, "y": y, "mask": mask, "path": str(path)}


def collate_cases(batch: List[Dict[str, Any]]):
    branch_inputs = np.stack([b["branch_input"] for b in batch], axis=0).astype(np.float32)
    y = np.stack([b["y"] for b in batch], axis=0).astype(np.float32)
    mask = np.stack([b["mask"] for b in batch], axis=0).astype(np.float32)
    paths = [b["path"] for b in batch]
    return {
        "branch_inputs": torch.from_numpy(branch_inputs),
        "y": torch.from_numpy(y),
        "mask": torch.from_numpy(mask),
        "paths": paths,
    }


# ------------------------------ Model ------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int, act: str = "relu"):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: List[nn.Module] = []
        d = in_dim
        if depth == 1:
            layers = [nn.Linear(d, out_dim)]
        else:
            for _ in range(depth - 1):
                layers.append(nn.Linear(d, width))
                layers.append(nn.ReLU(inplace=True) if act == "relu" else nn.GELU())
                d = width
            layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bihw,iohw->bohw", a, b)


class SpectralConv2d(nn.Module):
    """FNO spectral conv with safe AMP: force FFT to FP32 (autocast disabled in FFT block)."""

    def __init__(self, in_channels: int, out_channels: int, modes_z: int, modes_x: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_z = int(modes_z)
        self.modes_x = int(modes_x)

        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weights_pos = nn.Parameter(scale * torch.randn(self.in_channels, self.out_channels, self.modes_z, self.modes_x, dtype=torch.cfloat))
        self.weights_neg = nn.Parameter(scale * torch.randn(self.in_channels, self.out_channels, self.modes_z, self.modes_x, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        if x.is_cuda:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                x_f32 = x.float()
                x_ft = torch.fft.rfft2(x_f32, norm="ortho")

                out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)

                m1 = min(self.modes_z, H)
                m2 = min(self.modes_x, W // 2 + 1)

                out_ft[:, :, :m1, :m2]  = compl_mul2d(x_ft[:, :, :m1, :m2],  self.weights_pos[:, :, :m1, :m2])
                out_ft[:, :, -m1:, :m2] = compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights_neg[:, :, :m1, :m2])

                y = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        else:
            x_f32 = x.float()
            x_ft = torch.fft.rfft2(x_f32, norm="ortho")
            out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)
            m1 = min(self.modes_z, H)
            m2 = min(self.modes_x, W // 2 + 1)
            out_ft[:, :, :m1, :m2]  = compl_mul2d(x_ft[:, :, :m1, :m2],  self.weights_pos[:, :, :m1, :m2])
            out_ft[:, :, -m1:, :m2] = compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights_neg[:, :, :m1, :m2])
            y = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

        return y.to(dtype=x.dtype)


class FNOGridTrunk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        n_layers: int,
        modes_z: int,
        modes_x: int,
        padding_frac: float = 0.0,
    ):
        super().__init__()
        self.padding_frac = float(padding_frac)

        self.lift = nn.Conv2d(int(in_channels), int(width), kernel_size=1)
        self.spectral = nn.ModuleList([
            SpectralConv2d(int(width), int(width), int(modes_z), int(modes_x)) for _ in range(int(n_layers))
        ])
        self.pointwise = nn.ModuleList([
            nn.Conv2d(int(width), int(width), kernel_size=1) for _ in range(int(n_layers))
        ])
        self.act = nn.GELU()
        self.proj = nn.Sequential(
            nn.Conv2d(int(width), int(width), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(width), int(out_channels), kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = int(self.padding_frac * H)
        pad_w = int(self.padding_frac * W)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.lift(x)
        for k in range(len(self.spectral)):
            x = self.act(self.spectral[k](x) + self.pointwise[k](x))
        x = self.proj(x)

        if pad_h > 0 or pad_w > 0:
            x = x[..., :H, :W]
        return x


class DeepONetWindGridTrunkComplex(nn.Module):
    """forward(branch_inputs) -> (B,2,H,W) normalized target (p or Δp)."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        xg: np.ndarray,
        zg: np.ndarray,
        stats: Dict[str, Any],
        p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        super().__init__()
        self.cfg = dict(cfg)
        self.use_p0 = bool(cfg.get("use_p0_trunk", True))

        xg_t = torch.tensor(xg.astype(np.float32))
        zg_t = torch.tensor(zg.astype(np.float32))
        self.register_buffer("xg_buf", xg_t)
        self.register_buffer("zg_buf", zg_t)

        x_min, x_max = float(xg.min()), float(xg.max())
        z_min, z_max = float(zg.min()), float(zg.max())

        x_norm_1d = 2.0 * (xg_t - x_min) / (x_max - x_min + 1e-8) - 1.0
        z_norm_1d = 2.0 * (zg_t - z_min) / (z_max - z_min + 1e-8) - 1.0

        H = len(zg)
        W = len(xg)

        Xn = x_norm_1d[None, None, None, :].repeat(1, 1, H, 1)  # (1,1,H,W)
        Zn = z_norm_1d[None, None, :, None].repeat(1, 1, 1, W)  # (1,1,H,W)
        self.register_buffer("Xn", Xn)
        self.register_buffer("Zn", Zn)

        if self.use_p0:
            if p0 is None:
                raise ValueError("baseline p0 must be provided when use_p0_trunk=True")
            xg0, zg0, p0_re, p0_im = p0
            _assert_same_grid(xg, zg, xg0, zg0, "model_grid", "baseline_p0")

            pr = torch.tensor(p0_re.astype(np.float32))[None, None, :, :]
            pi = torch.tensor(p0_im.astype(np.float32))[None, None, :, :]

            if str(cfg.get("p0_feat_norm", "zscore")) == "zscore":
                pr = (pr - float(stats["p0_re_mean"])) / float(stats["p0_re_std"])
                pi = (pi - float(stats["p0_im_mean"])) / float(stats["p0_im_std"])

            self.register_buffer("p0_re_norm", pr)
            self.register_buffer("p0_im_norm", pi)
        else:
            self.register_buffer("p0_re_norm", torch.zeros(1, 1, H, W, dtype=torch.float32))
            self.register_buffer("p0_im_norm", torch.zeros(1, 1, H, W, dtype=torch.float32))

        L = int(cfg.get("latent", 64))
        self.L = L
        out_dim_lat = 2 * L

        m = int(cfg.get("branch_m", 128))
        use_dudz = bool(cfg.get("use_dudz", True))
        branch_in_dim = m * (2 if use_dudz else 1)

        self.branch = MLP(branch_in_dim, int(cfg.get("branch_width", 256)), int(cfg.get("branch_depth", 4)), out_dim_lat, act="relu")

        in_ch = 2  # x_norm, z_norm
        if self.use_p0:
            in_ch += 2

        self.trunk = FNOGridTrunk(
            in_channels=in_ch,
            out_channels=out_dim_lat,
            width=int(cfg.get("trunk_width", 64)),
            n_layers=int(cfg.get("trunk_layers", 4)),
            modes_z=int(cfg.get("modes_z", 24)),
            modes_x=int(cfg.get("modes_x", 64)),
            padding_frac=float(cfg.get("padding_frac", 0.05)),
        )

        self._printed_shapes = False

    def forward(self, branch_inputs: torch.Tensor) -> torch.Tensor:
        B = int(branch_inputs.shape[0])
        H = int(self.Zn.shape[2])
        W = int(self.Xn.shape[3])

        branch_inputs = branch_inputs.to(device=self.Xn.device)

        # branch -> (B,2,L)
        bvec = self.branch(branch_inputs).view(B, 2, self.L)

        feats: List[torch.Tensor] = [
            self.Xn.repeat(B, 1, 1, 1),
            self.Zn.repeat(B, 1, 1, 1),
        ]

        if self.use_p0:
            feats.append(self.p0_re_norm.repeat(B, 1, 1, 1))
            feats.append(self.p0_im_norm.repeat(B, 1, 1, 1))

        if bool(self.cfg.get("debug_shapes", False)) and (not self._printed_shapes):
            for i, f in enumerate(feats):
                print(f"[debug] trunk feat[{i}] shape={tuple(f.shape)}")
            print(f"[debug] branch_inputs {tuple(branch_inputs.shape)}")
            self._printed_shapes = True

        trunk_in = torch.cat(feats, dim=1)  # (B,C,H,W)
        tfeat = self.trunk(trunk_in).view(B, 2, self.L, H, W)
        out = torch.einsum("bcl,bclhw->bchw", bvec, tfeat)  # (B,2,H,W)
        return out


# ------------------------------ Loss / Metrics ------------------------------

def masked_mse_grid(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[:, None, :, :].to(pred.dtype)
    se = (pred - y) ** 2
    se = se * m
    denom = (m.sum() * pred.shape[1]).clamp_min(1e-8)
    return se.sum() / denom


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    L = max(1e-6, float(max(np.max(np.abs(a)), np.max(np.abs(b)))))
    c1 = (0.01 * L) ** 2; c2 = (0.03 * L) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2)
    return float(num / (den + 1e-12))


def _masked_error_stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    diff = (a - b)[mask]
    if diff.size == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae


# ------------------------------ Train / Eval ------------------------------

def _stats_compatible(stats: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    if bool(stats.get("predict_delta", False)) != bool(cfg.get("predict_delta", False)):
        return False
    if bool(stats.get("use_dudz", True)) != bool(cfg.get("use_dudz", True)):
        return False
    if bool(stats.get("normalize_u", True)) != bool(cfg.get("normalize_u", True)):
        return False
    if str(stats.get("u_norm_mode", "scale")) != str(cfg.get("u_norm_mode", "scale")):
        return False
    if float(stats.get("u_scale", 25.0)) != float(cfg.get("u_scale", 25.0)):
        return False
    if bool(cfg.get("predict_delta", False)):
        if str(stats.get("baseline_file", "")) != str(cfg.get("baseline_file", "")):
            return False
    if bool(stats.get("crop_to_safe_x_window", False)) != bool(cfg.get("crop_to_safe_x_window", False)):
        return False
    if bool(stats.get("use_p0_trunk", False)) != bool(cfg.get("use_p0_trunk", True)):
        return False
    return True


def train_run(
    data_root: Path,
    split: str,
    run_tag: str,
    lofo_family: str | None = None,
    device: str | None = None,
    cfg: Dict[str, Any] = CFG,
):
    cfg = dict(cfg)

    set_all_seeds(int(cfg["seed"]))
    torch.backends.cudnn.benchmark = True

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    man = load_manifest(data_root)
    if cfg.get("crop_to_safe_x_window", False) and "safe_x_window" in man:
        cfg["safe_x_window"] = tuple(man["safe_x_window"])

    splits_dir = data_root / "splits"
    if split == "iid":
        sp_dir = splits_dir / "iid"
    elif split == "lofo":
        assert lofo_family is not None
        sp_dir = splits_dir / "lofo" / lofo_family
    else:
        raise ValueError("split must be one of: iid, lofo")

    train_list = read_list(sp_dir / "train.txt")
    val_list   = read_list(sp_dir / "val.txt")

    if bool(cfg.get("debug_overfit", False)):
        n_dbg = max(1, min(int(cfg.get("debug_n_cases", 8)), len(train_list)))
        train_list = train_list[:n_dbg]
        val_list = list(train_list)
        print(f"[debug] OVERFIT mode ON -> using {len(train_list)} train/val cases")

    # establish grids (and optional crop) from first training case
    xg_full, zg = load_grids(train_list[0])

    idx_x = None
    xg = xg_full
    if cfg.get("crop_to_safe_x_window", False):
        xmin, xmax = cfg.get("safe_x_window", (-450.0, 450.0))
        idx_x = x_crop_indices(xg_full, float(xmin), float(xmax))
        if idx_x.size >= 16:
            xg = xg_full[idx_x]
            print(f"[train] Cropping x-grid to safe window: [{xmin},{xmax}] -> W={len(xg)}")
        else:
            idx_x = None
            print("[train] WARNING: safe window crop produced <16 points; disabling crop")

    # sensors_z saved under deeponet_aux
    aux_dir = data_root / "deeponet_aux"
    ensure_dir(aux_dir)

    sensors_path = aux_dir / f"sensors_z_m{cfg['branch_m']}.npy"
    if sensors_path.exists():
        sensors_z = np.load(sensors_path).astype(np.float32)
        print(f"[train] Loaded sensors_z from {sensors_path}")
    else:
        sensors_z = uniform_sensors_z(float(zg.min()), float(zg.max()), int(cfg["branch_m"]))
        np.save(sensors_path, sensors_z)
        print(f"[train] Wrote sensors_z to {sensors_path}")

    # baseline p0 (needed for predict_delta and/or use_p0_trunk)
    p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    if bool(cfg.get("predict_delta", False)) or bool(cfg.get("use_p0_trunk", True)):
        p0_full = load_baseline_p0(data_root, cfg)
        xg0, zg0, p0_re, p0_im = p0_full
        _assert_same_grid(xg_full, zg, xg0, zg0, "train_case0", "baseline_p0")
        if idx_x is not None:
            p0 = (xg0[idx_x], zg0, p0_re[:, idx_x], p0_im[:, idx_x])
        else:
            p0 = p0_full
        if bool(cfg.get("predict_delta", False)):
            print(f"[train] Using delta target (baseline_file={_baseline_path(data_root, cfg)})")
        if bool(cfg.get("use_p0_trunk", True)):
            print(f"[train] Trunk includes baseline p0 features (use_p0_trunk=True)")

    # stats path keyed by target and shear
    tgt_suffix = "delta" if bool(cfg.get("predict_delta", False)) else "p"
    shear_suffix = "shear" if bool(cfg.get("use_dudz", True)) else "noshear"
    stats_path = aux_dir / f"stats_wind_gridtrunk_{tgt_suffix}_{shear_suffix}_m{cfg['branch_m']}_L{cfg['latent']}.json"

    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if not _stats_compatible(stats, cfg):
            print("[train] Stats incompatible -> recomputing")
            stats = compute_stats_from_train(train_list, sensors_z, cfg, data_root=data_root, p0=p0)
            stats_path.write_text(json.dumps(stats, indent=2))
        else:
            print(f"[train] Loaded stats from {stats_path}")
    else:
        print("[train] Stats not found -> computing from train set (one-time cost)...")
        stats = compute_stats_from_train(train_list, sensors_z, cfg, data_root=data_root, p0=p0)
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"[train] Wrote stats to {stats_path}")

    # datasets
    tr_ds = WindDeepONetGridDataset(train_list, sensors_z, stats, cfg, data_root=data_root, p0=p0)
    va_ds = WindDeepONetGridDataset(val_list, sensors_z, stats, cfg, data_root=data_root, p0=p0)

    tr_ld = DataLoader(
        tr_ds,
        batch_size=int(cfg.get("batch_cases", 1)),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_cases,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=str(device).startswith("cuda"),
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=int(cfg.get("batch_cases", 1)),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_cases,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=str(device).startswith("cuda"),
    )

    model = DeepONetWindGridTrunkComplex(cfg, xg=xg, zg=zg, stats=stats, p0=p0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    use_autocast = bool(cfg.get("amp", False)) and str(device).startswith("cuda")
    has_complex_params = any(p.is_complex() for p in model.parameters())
    use_scaler = use_autocast and (not has_complex_params)
    if use_autocast and has_complex_params:
        print("[amp] Complex params detected -> GradScaler DISABLED (autocast-only).")

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_autocast else None
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    sched = None
    if bool(cfg.get("use_lr_scheduler", True)):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(cfg.get("sched_factor", 0.75)),
            patience=int(cfg.get("sched_patience", 10)),
            threshold=float(cfg.get("sched_threshold", 1e-5)),
            min_lr=float(cfg.get("sched_min_lr", 1e-6)),
            cooldown=int(cfg.get("sched_cooldown", 0)),
        )

    ckpt_dir = Path("checkpoints")
    ensure_dir(ckpt_dir)
    best_ckpt = ckpt_dir / f"{run_tag}.pt"
    last_ckpt = ckpt_dir / f"{run_tag}_last.pt"
    resume_path = best_ckpt if RESUME_FROM_BEST else last_ckpt

    best_val = float("inf")
    patience_left = int(cfg["patience"]) if bool(cfg.get("early_stop", False)) else 10**9
    start_epoch = 1

    def save_state(path: Path, epoch_i: int):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "epoch": epoch_i,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None,
            "sched": sched.state_dict() if (sched is not None) else None,
            "best_val": best_val,
            "patience_left": patience_left,

            # RNG states for deterministic resume
            "rng_torch": torch.get_rng_state(),
            "rng_cuda":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
            "rng_py":    random.getstate(),

            # metadata
            "cfg": cfg,
            "stats": stats,
            "sensors_z": sensors_z,
            "run_tag": run_tag,
            "split": split,
            "lofo_family": lofo_family,
            "x_grid": xg,
            "z_grid": zg,
        }, tmp)
        tmp.replace(path)

    # Resume
    if RESUME_TRAINING and resume_path.exists():
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            if ckpt.get("run_tag", run_tag) != run_tag:
                print(f"[resume] WARNING: checkpoint run_tag={ckpt.get('run_tag')} != current run_tag={run_tag}")

            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])

            if sched is not None and ckpt.get("sched") is not None:
                sched.load_state_dict(ckpt["sched"])

            if scaler is not None and scaler.is_enabled() and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])

            best_val = float(ckpt.get("best_val", best_val))
            patience_left = int(ckpt.get("patience_left", patience_left))

            if "rng_torch" in ckpt:
                torch.set_rng_state(ckpt["rng_torch"])
            if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
                torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
            if "rng_numpy" in ckpt:
                np.random.set_state(ckpt["rng_numpy"])
            if "rng_py" in ckpt:
                random.setstate(ckpt["rng_py"])

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"[resume] Loaded {resume_path} -> starting at epoch {start_epoch} (best_val={best_val:.6f})")
        except Exception as e:
            print(f"[resume] FAILED to load {resume_path} ({type(e).__name__}: {e}) -> starting fresh")
            start_epoch = 1
            best_val = float("inf")
            patience_left = int(cfg["patience"]) if bool(cfg.get("early_stop", False)) else 10**9
    else:
        if RESUME_TRAINING:
            print(f"[resume] No checkpoint found at {resume_path} -> starting fresh")

    print(f"[train] split={split} lofo={lofo_family} device={device_info_string(str(device))}")
    print(f"[train] Ntrain={len(train_list)} Nval={len(val_list)} batch_cases={cfg['batch_cases']} autocast={use_autocast} scaler={bool(scaler and scaler.is_enabled())}")
    print(f"[train] predict_delta={cfg.get('predict_delta', False)} baseline_file={cfg.get('baseline_file','')}")
    print(f"[train] sensors m={cfg.get('branch_m')} use_dudz={cfg.get('use_dudz', True)} normalize_u={cfg.get('normalize_u', True)} u_mode={cfg.get('u_norm_mode')} u_scale={cfg.get('u_scale')}")
    print(f"[train] trunk: use_p0_trunk={cfg.get('use_p0_trunk', True)} trunk_width={cfg.get('trunk_width')} layers={cfg.get('trunk_layers')} modes=({cfg.get('modes_z')},{cfg.get('modes_x')}) pad={cfg.get('padding_frac')}")

    last_epoch_ran = None
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        last_epoch_ran = epoch
        t0 = time.time()

        model.train()
        tr_loss = 0.0
        nb = 0
        for batch in tr_ld:
            branch_inputs = batch["branch_inputs"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.amp.autocast(device_type=device_type, enabled=True):
                    pred = model(branch_inputs)
                    loss = masked_mse_grid(pred, y, mask) if bool(cfg.get("masked_loss", False)) else torch.mean((pred - y) ** 2)

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            else:
                pred = model(branch_inputs)
                loss = masked_mse_grid(pred, y, mask) if bool(cfg.get("masked_loss", False)) else torch.mean((pred - y) ** 2)
                loss.backward()
                opt.step()

            tr_loss += float(loss.detach().cpu())
            nb += 1

        tr_loss /= max(1, nb)

        model.eval()
        va_loss = 0.0
        nb2 = 0
        with torch.no_grad():
            for batch in va_ld:
                branch_inputs = batch["branch_inputs"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                if use_autocast:
                    with torch.amp.autocast(device_type=device_type, enabled=True):
                        pred = model(branch_inputs)
                        loss = masked_mse_grid(pred, y, mask) if bool(cfg.get("masked_loss", False)) else torch.mean((pred - y) ** 2)
                else:
                    pred = model(branch_inputs)
                    loss = masked_mse_grid(pred, y, mask) if bool(cfg.get("masked_loss", False)) else torch.mean((pred - y) ** 2)

                va_loss += float(loss.detach().cpu())
                nb2 += 1

        va_loss /= max(1, nb2)

        if sched is not None:
            sched.step(va_loss)

        lr_now = float(opt.param_groups[0]["lr"])
        dt = time.time() - t0
        print(f"[epoch {epoch:04d}] train={tr_loss:.6f}  val={va_loss:.6f}  lr={lr_now:.2e}  ({dt:.1f}s)")

        improved = va_loss < (best_val - float(cfg.get("es_delta", 0.0)))
        if improved:
            best_val = va_loss
            if bool(cfg.get("early_stop", False)):
                patience_left = int(cfg["patience"])
            save_state(best_ckpt, epoch)
            print(f"  -> saved BEST checkpoint to {best_ckpt}")
        else:
            if bool(cfg.get("early_stop", False)):
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping (best val {best_val:.6f}).")
                    save_state(last_ckpt, epoch)
                    break

        save_state(last_ckpt, epoch)

    if last_epoch_ran is not None:
        save_state(last_ckpt, last_epoch_ran)

    print(f"[train] saved LAST checkpoint to {last_ckpt}")
    print(f"[train] best val {best_val:.6f} (best ckpt at {best_ckpt})")


@torch.no_grad()
def predict_case(
    model: nn.Module,
    device: str,
    path: Path,
    stats: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    data_root: Path,
    sensors_z: np.ndarray,
    p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns predicted physical p_re, p_im on the ROI grid (cropped if configured)."""

    cfg = dict(cfg)

    xg, zg = load_grids(path)

    idx_x = None
    if cfg.get("crop_to_safe_x_window", False):
        xmin, xmax = cfg.get("safe_x_window", (-450.0, 450.0))
        idx_x = x_crop_indices(xg, float(xmin), float(xmax))
        if idx_x.size >= 16:
            xg = xg[idx_x]
        else:
            idx_x = None

    # baseline
    if (bool(cfg.get("predict_delta", False)) or bool(cfg.get("use_p0_trunk", True))) and p0 is None:
        p0_full = load_baseline_p0(data_root, cfg)
        if idx_x is not None:
            xg0, zg0, p0_re, p0_im = p0_full
            p0 = (xg0[idx_x], zg0, p0_re[:, idx_x], p0_im[:, idx_x])
        else:
            p0 = p0_full

    # branch input for this case
    z_u, U = load_u_profile(path)
    if len(U) != len(zg):
        U = np.interp(zg.astype(np.float64), z_u.astype(np.float64), U.astype(np.float64)).astype(np.float32)

    z_range = float(zg.max() - zg.min() + 1e-8)

    U_s = np.interp(sensors_z.astype(np.float64), zg.astype(np.float64), U.astype(np.float64)).astype(np.float32)

    normalize_u = bool(cfg.get("normalize_u", True))
    u_norm_mode = str(cfg.get("u_norm_mode", "scale")).lower()
    u_scale = float(cfg.get("u_scale", 25.0))

    if not normalize_u:
        U_s_norm = U_s
    elif u_norm_mode == "zscore":
        mu = float(stats.get("u_mu", 0.0))
        sd = float(stats.get("u_std", 1.0))
        U_s_norm = (U_s - mu) / (sd + 1e-12)
    else:
        U_s_norm = U_s / (u_scale + 1e-12)

    parts = [U_s_norm.astype(np.float32)]

    if bool(cfg.get("use_dudz", True)):
        dudz = np.gradient(U.astype(np.float64), zg.astype(np.float64)).astype(np.float32)
        dudz_s = np.interp(sensors_z.astype(np.float64), zg.astype(np.float64), dudz.astype(np.float64)).astype(np.float32)

        if not normalize_u:
            dudz_s_norm = dudz_s
        elif u_norm_mode == "zscore":
            sd = float(stats.get("u_std", 1.0))
            dudz_s_norm = dudz_s * (z_range / (sd + 1e-12))
        else:
            dudz_s_norm = dudz_s * (z_range / (u_scale + 1e-12))

        parts.append(dudz_s_norm.astype(np.float32))

    bi = np.concatenate(parts, axis=0).astype(np.float32)
    bi_t = torch.from_numpy(bi[None, :]).to(device)

    # forward
    model.eval()
    pred_norm = model(bi_t).detach().cpu().numpy()[0]  # (2,H,W)

    # unnormalize
    y_re = pred_norm[0] * float(stats["p_re_std"]) + float(stats["p_re_mean"])
    y_im = pred_norm[1] * float(stats["p_im_std"]) + float(stats["p_im_mean"])

    # if predicting delta, reconstruct p = delta + p0
    if bool(cfg.get("predict_delta", False)):
        assert p0 is not None
        xg0, zg0, p0_re, p0_im = p0
        _assert_same_grid(xg, zg, xg0, zg0, str(path), "baseline_p0")
        y_re = y_re + p0_re
        y_im = y_im + p0_im

    return y_re.astype(np.float32), y_im.astype(np.float32)


def eval_run(
    data_root: Path,
    split: str,
    run_tag: str,
    *,
    ckpt: Path,
    device: str | None = None,
    lofo_family: str | None = None,
    cfg: Dict[str, Any] = CFG,
):
    cfg = dict(cfg)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    splits_dir = data_root / "splits"
    if split == "iid":
        sp_dir = splits_dir / "iid"
    elif split == "lofo":
        assert lofo_family is not None
        sp_dir = splits_dir / "lofo" / lofo_family
    else:
        raise ValueError("split must be one of: iid, lofo")

    test_list = read_list(sp_dir / "test.txt")

    # sensors
    aux_dir = data_root / "deeponet_aux"
    sensors_path = aux_dir / f"sensors_z_m{cfg['branch_m']}.npy"
    if not sensors_path.exists():
        raise FileNotFoundError(f"Missing sensors_z file: {sensors_path} (run training once to create it)")
    sensors_z = np.load(sensors_path).astype(np.float32)

    # baseline
    p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    if bool(cfg.get("predict_delta", False)) or bool(cfg.get("use_p0_trunk", True)):
        p0 = load_baseline_p0(data_root, cfg)

    # stats
    tgt_suffix = "delta" if bool(cfg.get("predict_delta", False)) else "p"
    shear_suffix = "shear" if bool(cfg.get("use_dudz", True)) else "noshear"
    stats_path = aux_dir / f"stats_wind_gridtrunk_{tgt_suffix}_{shear_suffix}_m{cfg['branch_m']}_L{cfg['latent']}.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats: {stats_path}")
    stats = json.loads(stats_path.read_text())

    state = torch.load(ckpt, map_location=device, weights_only=False)

    # model grid
    xg0, zg0 = load_grids(test_list[0])
    idx_x = None
    xg = xg0
    if cfg.get("crop_to_safe_x_window", False):
        xmin, xmax = cfg.get("safe_x_window", (-450.0, 450.0))
        idx_x = x_crop_indices(xg0, float(xmin), float(xmax))
        if idx_x.size >= 16:
            xg = xg0[idx_x]
            if p0 is not None:
                xgb, zgb, p0_re, p0_im = p0
                p0 = (xgb[idx_x], zgb, p0_re[:, idx_x], p0_im[:, idx_x])
        else:
            idx_x = None

    model = DeepONetWindGridTrunkComplex(cfg, xg=xg, zg=zg0, stats=stats, p0=p0).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    use_masked = bool(cfg.get("masked_metrics", False))

    all_rmse_re: List[float] = []
    all_rmse_im: List[float] = []
    all_mae_re: List[float] = []
    all_mae_im: List[float] = []
    all_ssim_re: List[float] = []
    all_ssim_im: List[float] = []

    per_family: Dict[str, Dict[str, List[float]]] = {}

    for path in test_list:
        # ground truth physical p
        xg_t, zg_t = load_grids(path)
        pre_raw, pim_raw = load_p_re_im(path)
        gt_re, gt_im = ensure_complex_zx(pre_raw, pim_raw, xg_t, zg_t)

        if idx_x is not None:
            gt_re = gt_re[:, idx_x]
            gt_im = gt_im[:, idx_x]
            xg_t = xg_t[idx_x]

        pr_re, pr_im = predict_case(
            model, device, path, stats, cfg, data_root=data_root, sensors_z=sensors_z, p0=p0
        )

        if use_masked:
            mask2d = np.ones_like(gt_re, dtype=bool)
            rmse_re, mae_re = _masked_error_stats(pr_re, gt_re, mask2d)
            rmse_im, mae_im = _masked_error_stats(pr_im, gt_im, mask2d)
            ssim_re = ssim_global(pr_re, gt_re)
            ssim_im = ssim_global(pr_im, gt_im)
        else:
            rmse_re = float(np.sqrt(np.mean((pr_re - gt_re) ** 2)))
            rmse_im = float(np.sqrt(np.mean((pr_im - gt_im) ** 2)))
            mae_re = float(np.mean(np.abs(pr_re - gt_re)))
            mae_im = float(np.mean(np.abs(pr_im - gt_im)))
            ssim_re = ssim_global(pr_re, gt_re)
            ssim_im = ssim_global(pr_im, gt_im)

        all_rmse_re.append(rmse_re); all_rmse_im.append(rmse_im)
        all_mae_re.append(mae_re); all_mae_im.append(mae_im)
        all_ssim_re.append(ssim_re); all_ssim_im.append(ssim_im)

        fam = case_family(path)
        d = per_family.setdefault(fam, {
            "rmse_re": [], "rmse_im": [],
            "mae_re": [], "mae_im": [],
            "ssim_re": [], "ssim_im": [],
        })
        d["rmse_re"].append(rmse_re); d["rmse_im"].append(rmse_im)
        d["mae_re"].append(mae_re); d["mae_im"].append(mae_im)
        d["ssim_re"].append(ssim_re); d["ssim_im"].append(ssim_im)

    def summarize(vals: List[float]) -> Dict[str, float]:
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
        }

    summary = {
        "overall": {
            "rmse_re": summarize(all_rmse_re),
            "rmse_im": summarize(all_rmse_im),
            "mae_re": summarize(all_mae_re),
            "mae_im": summarize(all_mae_im),
            "ssim_re": summarize(all_ssim_re),
            "ssim_im": summarize(all_ssim_im),
        },
        "per_family": {fam: {k: summarize(v) for k, v in d.items()} for fam, d in per_family.items()},
        "n_cases": len(test_list),
        "split": split,
        "run_tag": run_tag,
        "ckpt": str(ckpt),
        "masked_metrics": use_masked,
        "predict_delta": bool(cfg.get("predict_delta", False)),
        "baseline_file": str(cfg.get("baseline_file", "")) if bool(cfg.get("predict_delta", False)) else "",
        "normalize_u": bool(cfg.get("normalize_u", True)),
        "u_norm_mode": str(cfg.get("u_norm_mode", "scale")),
        "u_scale": float(cfg.get("u_scale", 25.0)),
        "use_dudz": bool(cfg.get("use_dudz", True)),
        "crop_to_safe_x_window": bool(cfg.get("crop_to_safe_x_window", False)),
        "deeponet": {
            "branch_m": int(cfg.get("branch_m", 128)),
            "latent": int(cfg.get("latent", 64)),
            "use_p0_trunk": bool(cfg.get("use_p0_trunk", True)),
            "trunk_width": int(cfg.get("trunk_width", 64)),
            "trunk_layers": int(cfg.get("trunk_layers", 4)),
            "modes_z": int(cfg.get("modes_z", 24)),
            "modes_x": int(cfg.get("modes_x", 64)),
            "padding_frac": float(cfg.get("padding_frac", 0.05)),
        },
    }

    out_dir = Path("evals")
    ensure_dir(out_dir)
    out_path = out_dir / f"{run_tag}__{split}{('_'+lofo_family) if split=='lofo' else ''}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[eval] wrote summary to {out_path}")


# ------------------------------ Autorun ------------------------------

def run_everything():
    if DATA_ROOT:
        data_root = Path(DATA_ROOT)
        print(f"[autorun] Using DATA_ROOT override: {data_root}")
        if not data_root.exists():
            data_root = find_data_root()
    else:
        data_root = find_data_root()

    if not (data_root / "splits" / "iid" / "train.txt").exists():
        print("[autorun] Splits not found -> creating...")
        make_splits(data_root)
    else:
        print("[autorun] Splits already present -> skipping make_splits.")

    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device_info_string(device)}")

    if RUN_IID:
        tag = RUN_TAG
        print(f"[autorun] Training IID as '{tag}'")
        train_run(data_root, split="iid", run_tag=tag, device=device)
        print(f"[autorun] Evaluating IID (BEST) as '{tag}'")
        eval_run(data_root, split="iid", run_tag=tag, ckpt=Path(f"checkpoints/{tag}.pt"), device=device)

    if RUN_LOFO_ALL:
        fams = sorted({case_family(p) for p in find_cases(data_root)})
        for fam in fams:
            tag = f"deeponet_wind_gridtrunk_fno_lofo_{fam}"
            print(f"[autorun] Training LOFO ({fam}) as '{tag}'")
            train_run(data_root, split="lofo", lofo_family=fam, run_tag=tag, device=device)
            print(f"[autorun] Evaluating LOFO ({fam}) as '{tag}'")
            eval_run(data_root, split="lofo", lofo_family=fam, run_tag=tag, ckpt=Path(f"checkpoints/{tag}.pt"), device=device)


if __name__ == "__main__":
    run_everything()
