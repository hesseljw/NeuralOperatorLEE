"""
Train a Fourier Neural Operator (FNO) surrogate for wind-only outdoor acoustics (LEE data).

Model: grid-to-grid FNO that consumes coordinate features (and optional wind features) on the ROI grid and predicts
complex pressure (or residual Δp) on the same grid.

Outputs are saved under runs/<RUN_TAG>/ (checkpoints, metrics, eval summaries).

Quickstart:
  python train_fno_wind_coords.py --data-root <DATA_DIR> --iid --device cuda
"""

from __future__ import annotations

import json
import time
import random
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
DEVICE    = None   # "cuda" or "cpu" (if None, auto-pick)
RUN_TAG   = "fno_wind_u_coords_learn_p_iid"

RUN_IID      = True
RUN_LOFO_ALL = False

CFG: Dict[str, Any] = {
    "seed": 1234,

    # ---- Training ----
    "epochs": 4000,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 1,
    "num_workers": 0,
    "amp": True,

    # ---- Early stopping (optional) ----
    "early_stop": False,
    "patience": 40,
    "es_delta": 1e-4,

    # ---- Save ----
    "save_every": 1,

    # ---- Masking flags (kept for parity; wind ROI is typically all-air) ----
    "masked_loss": False,
    "masked_stats": False,
    "masked_metrics": False,

    # ---- p vs delta ----
    # If True: learn delta to baseline p0 (flat ground, no wind): \Delta p = p - p0
    # If False: learn p directly.
    "predict_delta": False,
    "baseline_file": "deeponet_aux/flat_ground.h5",  # relative to DATA_ROOT

    # ---- Optional crop to safe window ----
    "crop_to_safe_x_window": False,
    "safe_x_window": (-450.0, 450.0),

    # ---- Debug ----
    "debug_overfit": False,
    "debug_n_cases": 8,
    "debug_constant_target": False,
    "debug_constant_re": 0.0,
    "debug_constant_im": 0.0,

    # ---- Wind input features ----
    "use_dudz": False,

    # ---- Wind normalization ----
    # mode: "scale" uses U_norm=U/u_scale (default u_scale=25 from generator)
    #       "zscore" uses U_norm=(U-mu)/std over training set (mu/std stored in stats)
    "normalize_u": True,
    "u_norm_mode": "scale",
    "u_scale": 25.0,

    # ---- FNO architecture ----
    # in_channels is inferred from use_dudz (3 or 4) to avoid config mismatch.
    "in_channels": 3,
    "out_channels": 2,
    "width": 64,
    "n_layers": 4,
    "modes_z": 24,
    "modes_x": 64,
    "padding_frac": 0.05,

    # ---- LR Scheduler (ReduceLROnPlateau) ----
    "use_lr_scheduler": True,
    "sched_factor": 0.75,
    "sched_patience": 10,
    "sched_threshold": 1e-5,
    "sched_min_lr": 1e-6,
    "sched_cooldown": 0,
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


# ------------------------------ Splits (same style as terrain) ------------------------------

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


# ------------------------------ Stats ------------------------------

def compute_stats_from_train(
    train_list: List[Path],
    cfg: Dict[str, Any],
    *,
    data_root: Path,
    p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    predict_delta = bool(cfg.get("predict_delta", False))
    if predict_delta and p0 is None:
        p0 = load_baseline_p0(data_root, cfg)

    re_vals: List[np.ndarray] = []
    im_vals: List[np.ndarray] = []

    u_vals: List[np.ndarray] = []

    for p in train_list:
        xg, zg = load_grids(p)
        pre_raw, pim_raw = load_p_re_im(p)
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
            _assert_same_grid(xg, zg, xg0, zg0, str(p), "baseline_p0")
            dpre = pre - p0_re
            dpim = pim - p0_im
        else:
            dpre, dpim = pre, pim

        re_vals.append(dpre.reshape(-1).astype(np.float32))
        im_vals.append(dpim.reshape(-1).astype(np.float32))

        if bool(cfg.get("normalize_u", True)) and str(cfg.get("u_norm_mode", "scale")).lower() == "zscore":
            _, U = load_u_profile(p)
            u_vals.append(U.reshape(-1).astype(np.float32))

    re_cat = np.concatenate(re_vals) if re_vals else np.zeros((0,), dtype=np.float32)
    im_cat = np.concatenate(im_vals) if im_vals else np.zeros((0,), dtype=np.float32)

    u_mu, u_std = 0.0, 1.0
    if u_vals:
        u_cat = np.concatenate(u_vals)
        u_mu = float(u_cat.mean())
        u_std = float(u_cat.std() + 1e-8)

    return {
        "p_re_mean": float(re_cat.mean()) if re_cat.size else 0.0,
        "p_re_std":  float(re_cat.std() + 1e-8) if re_cat.size else 1.0,
        "p_im_mean": float(im_cat.mean()) if im_cat.size else 0.0,
        "p_im_std":  float(im_cat.std() + 1e-8) if im_cat.size else 1.0,
        "predict_delta": predict_delta,
        "baseline_file": str(cfg.get("baseline_file", "")) if predict_delta else "",
        "normalize_u": bool(cfg.get("normalize_u", True)),
        "u_norm_mode": str(cfg.get("u_norm_mode", "scale")),
        "u_scale": float(cfg.get("u_scale", 25.0)),
        "u_mu": u_mu,
        "u_std": u_std,
        "use_dudz": bool(cfg.get("use_dudz", False)),
        "crop_to_safe_x_window": bool(cfg.get("crop_to_safe_x_window", False)),
        "safe_x_window": tuple(cfg.get("safe_x_window", (-450.0, 450.0))),
    }


# ------------------------------ Dataset ------------------------------

class WindFNOComplexDataset(Dataset):
    def __init__(
        self,
        paths: List[Path],
        stats: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        data_root: Path,
        p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        self.paths = paths
        self.stats = stats
        self.cfg = cfg
        self.data_root = data_root
        self.predict_delta = bool(cfg.get("predict_delta", False))
        self.use_dudz = bool(cfg.get("use_dudz", False))
        self.normalize_u = bool(cfg.get("normalize_u", True))
        self.u_norm_mode = str(cfg.get("u_norm_mode", "scale")).lower()
        self.u_scale = float(cfg.get("u_scale", 25.0))

        self.p0 = p0
        if self.predict_delta and self.p0 is None:
            self.p0 = load_baseline_p0(data_root, cfg)

    def __len__(self):
        return len(self.paths)

    def _normalize_u_1d(self, U: np.ndarray) -> np.ndarray:
        if not self.normalize_u:
            return U.astype(np.float32)
        if self.u_norm_mode == "zscore":
            mu = float(self.stats.get("u_mu", 0.0))
            sd = float(self.stats.get("u_std", 1.0))
            return ((U - mu) / sd).astype(np.float32)
        # scale
        return (U / (self.u_scale + 1e-12)).astype(np.float32)

    def _dudz_norm_1d(self, dudz: np.ndarray, z_range: float) -> np.ndarray:
        if not self.normalize_u:
            return dudz.astype(np.float32)
        if self.u_norm_mode == "zscore":
            sd = float(self.stats.get("u_std", 1.0))
            return (dudz * (z_range / (sd + 1e-12))).astype(np.float32)
        # scale
        return (dudz * (z_range / (self.u_scale + 1e-12))).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.paths[idx]

        xg, zg = load_grids(p)
        pre_raw, pim_raw = load_p_re_im(p)
        pre, pim = ensure_complex_zx(pre_raw, pim_raw, xg, zg)

        idx_x = None
        if self.cfg.get("crop_to_safe_x_window", False):
            xmin, xmax = self.cfg.get("safe_x_window", (-450.0, 450.0))
            idx_x = x_crop_indices(xg, float(xmin), float(xmax))
            if idx_x.size >= 16:
                xg = xg[idx_x]
                pre = pre[:, idx_x]
                pim = pim[:, idx_x]

        # target (p or delta)
        if self.predict_delta:
            assert self.p0 is not None
            xg0, zg0, p0_re, p0_im = self.p0
            if idx_x is not None and idx_x.size >= 16:
                xg0 = xg0[idx_x]
                p0_re = p0_re[:, idx_x]
                p0_im = p0_im[:, idx_x]
            _assert_same_grid(xg, zg, xg0, zg0, str(p), "baseline_p0")
            dpre = pre - p0_re
            dpim = pim - p0_im
        else:
            dpre, dpim = pre, pim

        H, W = dpre.shape

        if self.cfg.get("debug_constant_target", False):
            cre = float(self.cfg.get("debug_constant_re", 0.0))
            cim = float(self.cfg.get("debug_constant_im", 0.0))
            Y = np.stack([
                np.full((H, W), cre, np.float32),
                np.full((H, W), cim, np.float32),
            ], axis=0)
        else:
            y_re = (dpre - float(self.stats["p_re_mean"])) / float(self.stats["p_re_std"])
            y_im = (dpim - float(self.stats["p_im_mean"])) / float(self.stats["p_im_std"])
            Y = np.stack([y_re, y_im], axis=0).astype(np.float32)

        # wind profile -> 2D channels
        z_u, U = load_u_profile(p)
        if len(U) != len(zg):
            # robust fallback: interpolate U(z) to ROI z_grid
            U = np.interp(zg.astype(np.float64), z_u.astype(np.float64), U.astype(np.float64)).astype(np.float32)

        U_norm_1d = self._normalize_u_1d(U)
        U2d = np.broadcast_to(U_norm_1d[:, None], (H, W)).astype(np.float32)

        z_range = float(zg.max() - zg.min() + 1e-8)

        channels: List[np.ndarray] = [U2d]

        if self.use_dudz:
            dudz = np.gradient(U.astype(np.float64), zg.astype(np.float64)).astype(np.float32)
            dudz_norm_1d = self._dudz_norm_1d(dudz, z_range)
            dudz2d = np.broadcast_to(dudz_norm_1d[:, None], (H, W)).astype(np.float32)
            channels.append(dudz2d)

        # coord channels in [-1,1]
        x_norm_1d = (2.0 * (xg - xg.min()) / (xg.max() - xg.min() + 1e-8) - 1.0).astype(np.float32)
        z_norm_1d = (2.0 * (zg - zg.min()) / (zg.max() - zg.min() + 1e-8) - 1.0).astype(np.float32)
        x_norm = np.broadcast_to(x_norm_1d[None, :], (H, W))
        z_norm = np.broadcast_to(z_norm_1d[:, None], (H, W))

        channels += [x_norm.astype(np.float32), z_norm.astype(np.float32)]
        X = np.stack(channels, axis=0).astype(np.float32)

        mask = np.ones((H, W), dtype=np.float32)

        return {"X": X, "Y": Y, "mask": mask, "path": str(p)}


def collate_fields(batch: List[Dict[str, Any]]):
    X = torch.from_numpy(np.stack([b["X"] for b in batch], axis=0))
    Y = torch.from_numpy(np.stack([b["Y"] for b in batch], axis=0))
    M = torch.from_numpy(np.stack([b["mask"] for b in batch], axis=0))
    paths = [b["path"] for b in batch]
    return {"X": X, "Y": Y, "mask": M, "paths": paths}


# ------------------------------ FNO model (copied from terrain) ------------------------------

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_z: int, modes_x: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_z = modes_z
        self.modes_x = modes_x

        scale = 1.0 / (in_channels * out_channels)
        self.weight_pos = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_z, modes_x, 2))
        self.weight_neg = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_z, modes_x, 2))

    @staticmethod
    def _compl_mul2d(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        wc = torch.complex(w[..., 0], w[..., 1]).to(a.dtype)
        return torch.einsum("bixy,ioxy->boxy", a, wc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        orig_dtype = x.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            x_fft = x.float()
        else:
            x_fft = x

        x_ft = torch.fft.rfft2(x_fft, norm="ortho")
        Wf = x_ft.shape[-1]
        out_ft = torch.zeros(B, self.out_channels, H, Wf, dtype=torch.complex64, device=x.device)

        mz = min(self.modes_z, H)
        mx = min(self.modes_x, Wf)

        out_ft[:, :, :mz, :mx]  = self._compl_mul2d(x_ft[:, :, :mz, :mx],  self.weight_pos[:, :, :mz, :mx, :])
        out_ft[:, :, -mz:, :mx] = self._compl_mul2d(x_ft[:, :, -mz:, :mx], self.weight_neg[:, :, :mz, :mx, :])

        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        if orig_dtype in (torch.float16, torch.bfloat16):
            x_out = x_out.to(orig_dtype)
        return x_out


class FNO2d(nn.Module):
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

        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes_z, modes_x) for _ in range(n_layers)])
        self.w1x1 = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.InstanceNorm2d(width, affine=True) for _ in range(n_layers)])

        self.proj1 = nn.Conv2d(width, width, kernel_size=1)
        self.proj2 = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.lift(x)

        pad_h = int(self.padding_frac * H)
        pad_w = int(self.padding_frac * W)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        for spec, w, n in zip(self.spec, self.w1x1, self.norm):
            x = spec(x) + w(x)
            x = n(x)
            x = F.gelu(x)

        x = F.gelu(self.proj1(x))
        x = self.proj2(x)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x


# ------------------------------ Loss / Metrics ------------------------------

def masked_mse_field(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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

def _infer_in_channels(cfg: Dict[str, Any]) -> int:
    return 4 if bool(cfg.get("use_dudz", False)) else 3


def train_run(
    data_root: Path,
    split: str,
    run_tag: str,
    lofo_family: str | None = None,
    device: str | None = None,
    cfg=CFG,
):
    cfg = dict(cfg)
    cfg["in_channels"] = _infer_in_channels(cfg)

    set_all_seeds(int(cfg["seed"]))

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

    if cfg.get("debug_overfit", False):
        n_dbg = int(cfg.get("debug_n_cases", 8))
        n_dbg = max(1, min(n_dbg, len(train_list)))
        train_list = train_list[:n_dbg]
        val_list = list(train_list)
        print(f"[debug] OVERFIT mode ON -> using {len(train_list)} train/val cases")

    if cfg.get("debug_constant_target", False):
        print(f"[debug] CONSTANT TARGET mode ON -> y_re={cfg.get('debug_constant_re',0.0)} y_im={cfg.get('debug_constant_im',0.0)} (normalized)")

    p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    if bool(cfg.get("predict_delta", False)):
        p0 = load_baseline_p0(data_root, cfg)
        xg0, zg0, _, _ = p0
        xg1, zg1 = load_grids(train_list[0])
        _assert_same_grid(xg1, zg1, xg0, zg0, str(train_list[0]), "baseline_p0")
        print(f"[train] Using delta target (baseline_file={_baseline_path(data_root, cfg)})")

    aux_dir = data_root / "fno_aux"
    ensure_dir(aux_dir)
    tgt_suffix = "delta" if bool(cfg.get("predict_delta", False)) else "p"
    dudz_suffix = "dudz" if bool(cfg.get("use_dudz", False)) else "u"
    stats_path = aux_dir / f"stats_complex_wind_{tgt_suffix}_{dudz_suffix}.json"

    def _stats_compatible(st: Dict[str, Any], cfg_local: Dict[str, Any]) -> bool:
        if bool(st.get("predict_delta", False)) != bool(cfg_local.get("predict_delta", False)):
            return False
        if bool(st.get("use_dudz", False)) != bool(cfg_local.get("use_dudz", False)):
            return False
        if bool(st.get("normalize_u", True)) != bool(cfg_local.get("normalize_u", True)):
            return False
        if str(st.get("u_norm_mode", "scale")) != str(cfg_local.get("u_norm_mode", "scale")):
            return False
        if float(st.get("u_scale", 25.0)) != float(cfg_local.get("u_scale", 25.0)):
            return False
        if bool(cfg_local.get("predict_delta", False)):
            if str(st.get("baseline_file", "")) != str(cfg_local.get("baseline_file", "")):
                return False
        if bool(st.get("crop_to_safe_x_window", False)) != bool(cfg_local.get("crop_to_safe_x_window", False)):
            return False
        return True

    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if not _stats_compatible(stats, cfg):
            print("[train] Stats incompatible -> recomputing")
            stats = compute_stats_from_train(train_list, cfg, data_root=data_root, p0=p0)
            stats_path.write_text(json.dumps(stats, indent=2))
        else:
            print(f"[train] Loaded stats from {stats_path}")
    else:
        stats = compute_stats_from_train(train_list, cfg, data_root=data_root, p0=p0)
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"[train] Wrote stats to {stats_path}")

    tr_ds = WindFNOComplexDataset(train_list, stats, cfg, data_root=data_root, p0=p0)
    va_ds = WindFNOComplexDataset(val_list, stats, cfg, data_root=data_root, p0=p0)

    tr_ld = DataLoader(
        tr_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_fields,
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_fields,
    )

    model = FNO2d(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        width=int(cfg["width"]),
        n_layers=int(cfg["n_layers"]),
        modes_z=int(cfg["modes_z"]),
        modes_x=int(cfg["modes_x"]),
        padding_frac=float(cfg.get("padding_frac", 0.0)),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    use_amp = bool(cfg.get("amp", False)) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    ckpt_dir = Path("checkpoints")
    ensure_dir(ckpt_dir)
    best_ckpt = ckpt_dir / f"{run_tag}.pt"
    last_ckpt = ckpt_dir / f"{run_tag}_last.pt"

    best_val = float("inf")
    patience_left = int(cfg["patience"]) if cfg.get("early_stop", False) else 10**9

    sched = None

    def save_state(path: Path, epoch_i: int):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "epoch": epoch_i,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if (use_amp and scaler is not None) else None,
            "sched": sched.state_dict() if (sched is not None) else None,
            "best_val": best_val,
            "patience_left": patience_left,
            "cfg": cfg,
            "stats": stats,
            "run_tag": run_tag,
            "split": split,
            "lofo_family": lofo_family,
        }, tmp)
        tmp.replace(path)

    start_epoch = 1
    resume_path = last_ckpt if last_ckpt.exists() else (best_ckpt if best_ckpt.exists() else None)
    resume_state = None
    if resume_path is not None:
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"])
        if resume_state.get("opt") is not None:
            opt.load_state_dict(resume_state["opt"])
        if use_amp and scaler is not None and resume_state.get("scaler") is not None:
            try:
                scaler.load_state_dict(resume_state["scaler"])
            except Exception:
                pass
        best_val = float(resume_state.get("best_val", best_val))
        patience_left = int(resume_state.get("patience_left", patience_left))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        print(f"[train] Resumed from {resume_path} -> starting at epoch {start_epoch}")

    if bool(cfg.get("use_lr_scheduler", True)):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(cfg.get("sched_factor", 0.5)),
            patience=int(cfg.get("sched_patience", 25)),
            threshold=float(cfg.get("sched_threshold", 1e-5)),
            min_lr=float(cfg.get("sched_min_lr", 1e-6)),
            cooldown=int(cfg.get("sched_cooldown", 0)),
        )
        if resume_state is not None and resume_state.get("sched") is not None:
            try:
                sched.load_state_dict(resume_state["sched"])
            except Exception:
                pass

    print(f"[train] split={split} lofo={lofo_family} device={device_info_string(device)}")
    print(f"[train] Ntrain={len(train_list)} Nval={len(val_list)} batch={cfg['batch_size']} amp={use_amp}")
    print(f"[train] masked_loss={cfg.get('masked_loss', False)} masked_stats={cfg.get('masked_stats', False)} masked_metrics={cfg.get('masked_metrics', False)}")
    print(f"[train] predict_delta={cfg.get('predict_delta', False)} baseline_file={cfg.get('baseline_file', '')}")
    print(f"[train] normalize_u={cfg.get('normalize_u', True)} u_norm_mode={cfg.get('u_norm_mode')} u_scale={cfg.get('u_scale')} use_dudz={cfg.get('use_dudz', False)}")
    print(f"[train] crop_to_safe_x_window={cfg.get('crop_to_safe_x_window', False)} safe_x_window={cfg.get('safe_x_window')}")
    print(f"[train] FNO: in_ch={cfg['in_channels']} width={cfg['width']} layers={cfg['n_layers']} modes(z,x)=({cfg['modes_z']},{cfg['modes_x']}) padding_frac={cfg.get('padding_frac',0.0)}")
    if sched is not None:
        print(f"[train] LR scheduler: ReduceLROnPlateau(factor={cfg.get('sched_factor')}, patience={cfg.get('sched_patience')}, min_lr={cfg.get('sched_min_lr')})")

    save_every = max(1, int(cfg.get("save_every", 1)))
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    last_epoch_ran = None
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        last_epoch_ran = epoch
        t0 = time.time()

        model.train()
        tr_loss = 0.0
        nb = 0
        for batch in tr_ld:
            X = batch["X"].to(device, non_blocking=True)
            Y = batch["Y"].to(device, non_blocking=True)
            M = batch["mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                assert scaler is not None
                with torch.amp.autocast(device_type=device_type, enabled=True):
                    pred = model(X)
                    loss = masked_mse_field(pred, Y, M) if cfg.get("masked_loss", False) else torch.mean((pred - Y) ** 2)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(X)
                loss = masked_mse_field(pred, Y, M) if cfg.get("masked_loss", False) else torch.mean((pred - Y) ** 2)
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
                X = batch["X"].to(device, non_blocking=True)
                Y = batch["Y"].to(device, non_blocking=True)
                M = batch["mask"].to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast(device_type=device_type, enabled=True):
                        pred = model(X)
                        loss = masked_mse_field(pred, Y, M) if cfg.get("masked_loss", False) else torch.mean((pred - Y) ** 2)
                else:
                    pred = model(X)
                    loss = masked_mse_field(pred, Y, M) if cfg.get("masked_loss", False) else torch.mean((pred - Y) ** 2)

                va_loss += float(loss.detach().cpu())
                nb2 += 1
        va_loss /= max(1, nb2)

        if sched is not None:
            sched.step(va_loss)

        dt = time.time() - t0
        lr_now = float(opt.param_groups[0]["lr"])
        print(f"[epoch {epoch:04d}] train={tr_loss:.6f}  val={va_loss:.6f}  lr={lr_now:.2e}  ({dt:.1f}s)")

        improved = va_loss < (best_val - float(cfg.get("es_delta", 0.0)))
        if improved:
            best_val = va_loss
            if cfg.get("early_stop", False):
                patience_left = int(cfg["patience"])
            save_state(best_ckpt, epoch)
            print(f"  -> saved BEST checkpoint to {best_ckpt}")
        else:
            if cfg.get("early_stop", False):
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping (best val {best_val:.6f}).")
                    save_state(last_ckpt, epoch)
                    break

        if (epoch % save_every) == 0:
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
    p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = dict(cfg)
    cfg["in_channels"] = _infer_in_channels(cfg)

    xg, zg = load_grids(path)

    if bool(cfg.get("predict_delta", False)) and p0 is None:
        p0 = load_baseline_p0(data_root, cfg)

    idx_x = None
    if cfg.get("crop_to_safe_x_window", False):
        xmin, xmax = cfg.get("safe_x_window", (-450.0, 450.0))
        idx_x = x_crop_indices(xg, float(xmin), float(xmax))
        if idx_x.size >= 16:
            xg = xg[idx_x]
            if p0 is not None:
                xg0, zg0, p0_re, p0_im = p0
                p0 = (xg0[idx_x], zg0, p0_re[:, idx_x], p0_im[:, idx_x])

    if p0 is not None:
        xg0, zg0, _, _ = p0
        _assert_same_grid(xg, zg, xg0, zg0, str(path), "baseline_p0")

    z_u, U = load_u_profile(path)
    if len(U) != len(zg):
        U = np.interp(zg.astype(np.float64), z_u.astype(np.float64), U.astype(np.float64)).astype(np.float32)

    normalize_u = bool(cfg.get("normalize_u", True))
    u_norm_mode = str(cfg.get("u_norm_mode", "scale")).lower()
    u_scale = float(cfg.get("u_scale", 25.0))

    if not normalize_u:
        U_norm_1d = U.astype(np.float32)
    elif u_norm_mode == "zscore":
        mu = float(stats.get("u_mu", 0.0))
        sd = float(stats.get("u_std", 1.0))
        U_norm_1d = ((U - mu) / (sd + 1e-12)).astype(np.float32)
    else:
        U_norm_1d = (U / (u_scale + 1e-12)).astype(np.float32)

    H, W = len(zg), len(xg)
    U2d = np.broadcast_to(U_norm_1d[:, None], (H, W)).astype(np.float32)

    z_range = float(zg.max() - zg.min() + 1e-8)

    channels: List[np.ndarray] = [U2d]

    if bool(cfg.get("use_dudz", False)):
        dudz = np.gradient(U.astype(np.float64), zg.astype(np.float64)).astype(np.float32)
        if not normalize_u:
            dudz_norm_1d = dudz.astype(np.float32)
        elif u_norm_mode == "zscore":
            sd = float(stats.get("u_std", 1.0))
            dudz_norm_1d = (dudz * (z_range / (sd + 1e-12))).astype(np.float32)
        else:
            dudz_norm_1d = (dudz * (z_range / (u_scale + 1e-12))).astype(np.float32)
        channels.append(np.broadcast_to(dudz_norm_1d[:, None], (H, W)).astype(np.float32))

    x_norm_1d = (2.0 * (xg - xg.min()) / (xg.max() - xg.min() + 1e-8) - 1.0).astype(np.float32)
    z_norm_1d = (2.0 * (zg - zg.min()) / (zg.max() - zg.min() + 1e-8) - 1.0).astype(np.float32)
    x_norm = np.broadcast_to(x_norm_1d[None, :], (H, W)).astype(np.float32)
    z_norm = np.broadcast_to(z_norm_1d[:, None], (H, W)).astype(np.float32)

    channels += [x_norm, z_norm]

    X = np.stack(channels, axis=0)[None, ...].astype(np.float32)
    X_t = torch.from_numpy(X).to(device)

    pred_std = model(X_t).float().cpu().numpy()[0]

    y_re = pred_std[0] * float(stats["p_re_std"]) + float(stats["p_re_mean"])
    y_im = pred_std[1] * float(stats["p_im_std"]) + float(stats["p_im_mean"])

    if bool(cfg.get("predict_delta", False)):
        assert p0 is not None
        _, _, p0_re, p0_im = p0
        y_re = y_re + p0_re
        y_im = y_im + p0_im

    return y_re.astype(np.float32), y_im.astype(np.float32)


def eval_run(
    data_root: Path,
    split: str,
    run_tag: str,
    ckpt: Path,
    lofo_family: str | None = None,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = state.get("cfg", CFG)
    cfg = dict(cfg)
    cfg["in_channels"] = _infer_in_channels(cfg)

    stats = state["stats"]

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

    test_list = read_list(sp_dir / "test.txt")

    model = FNO2d(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        width=int(cfg["width"]),
        n_layers=int(cfg["n_layers"]),
        modes_z=int(cfg["modes_z"]),
        modes_x=int(cfg["modes_x"]),
        padding_frac=float(cfg.get("padding_frac", 0.0)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    p0 = None
    if bool(cfg.get("predict_delta", False)):
        p0 = load_baseline_p0(data_root, cfg)
        print(f"[eval] Using baseline reconstruction (baseline_file={_baseline_path(data_root, cfg)})")

    use_masked = bool(cfg.get("masked_metrics", False))

    all_rmse_re, all_rmse_im = [], []
    all_mae_re, all_mae_im = [], []
    all_ssim_re, all_ssim_im = [], []

    per_family: Dict[str, Dict[str, List[float]]] = {}

    for path in test_list:
        xg, zg = load_grids(path)
        gt_re_raw, gt_im_raw = load_p_re_im(path)
        gt_re, gt_im = ensure_complex_zx(gt_re_raw, gt_im_raw, xg, zg)

        if cfg.get("crop_to_safe_x_window", False):
            xmin, xmax = cfg.get("safe_x_window", (-450.0, 450.0))
            idx_x = x_crop_indices(xg, float(xmin), float(xmax))
            if idx_x.size >= 16:
                xg = xg[idx_x]
                gt_re = gt_re[:, idx_x]
                gt_im = gt_im[:, idx_x]

        pr_re, pr_im = predict_case(model, device, path, stats, cfg, data_root=data_root, p0=p0)

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
        "use_dudz": bool(cfg.get("use_dudz", False)),
        "crop_to_safe_x_window": bool(cfg.get("crop_to_safe_x_window", False)),
        "fno": {
            "in_channels": cfg.get("in_channels"),
            "width": cfg.get("width"),
            "n_layers": cfg.get("n_layers"),
            "modes_z": cfg.get("modes_z"),
            "modes_x": cfg.get("modes_x"),
            "padding_frac": cfg.get("padding_frac"),
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
            tag = f"fno_wind_u_coords_p_or_delta_lofo_{fam}"
            print(f"[autorun] Training LOFO ({fam}) as '{tag}'")
            train_run(data_root, split="lofo", lofo_family=fam, run_tag=tag, device=device)
            print(f"[autorun] Evaluating LOFO ({fam}) as '{tag}'")
            eval_run(data_root, split="lofo", lofo_family=fam, run_tag=tag, ckpt=Path(f"checkpoints/{tag}.pt"), device=device)


if __name__ == "__main__":
    run_everything()
