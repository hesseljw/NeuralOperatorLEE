"""
Train a DeepONet-style surrogate for terrain-only outdoor acoustics (LEE data).

Model: branch network encodes terrain h(x) via sensors; trunk uses grid coordinates (x,z) with optional η=z-h(x)
and Fourier features, merged via a dot-product to predict complex pressure on the ROI grid.

Target:
- residual Δp(x,z)=p_terrain(x,z)-p_flat(x,z) (reconstructed as p = p_flat + Δp)
Loss/metrics are masked to the air region (z >= h(x)).

Outputs are saved under runs/<RUN_TAG>/ (checkpoints, metrics, eval summaries).

Quickstart:
  python train_deeponet_terrain_eta_fourier.py --data-root <DATA_DIR> --iid --device cuda
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
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# USER CONFIG
# =============================================================================
# Edit the values below for your machine / experiment. You can also override many
# of them via command-line flags (see --help).

DATA_ROOT = "data/terrain_sobol_complex_n_1000"
DEVICE    = None  # "cuda" or "cpu" (if None auto)
RUN_TAG   = "deeponet_gridtrunk_fno_learn_p_iid"

RUN_IID        = True
RUN_LOFO_ALL   = False
RUN_RANGE_OOD  = False

# ------------------------------ Resume toggles ------------------------------
RESUME_TRAINING = True          # if True, resume from *_last.pt when available
RESUME_FROM_BEST = False        # if True, resume from best checkpoint instead of last (usually keep False)


CFG: Dict[str, Any] = {
    "seed": 1234,

    # ---- Optimization ----
    "epochs": 1500,
    "lr": 1e-4,
    "weight_decay": 1e-5,

    "early_stop": False,
    "patience": 30,
    "es_delta": 1e-5,

    # ---- AMP ----
    "amp": True,  # autocast ON, GradScaler auto-disabled if complex params exist

    # ---- LR scheduler ----
    "use_lr_scheduler": True,
    "sched_factor": 0.75,
    "sched_patience": 12,
    "sched_threshold": 1e-5,
    "sched_min_lr": 1e-6,
    "sched_cooldown": 0,

    # ---- Batch ----
    "batch_cases": 1,      # grid trunk heavy
    "num_workers": 0,      # keep 0 with HDF5

    # ---- Branch sensors ----
    "branch_m": 500,

    # ---- Normalization ----
    "H_scale": 100.0,

    # ---- Masking ----
    "masked_loss": True,
    "masked_stats": True,
    "masked_metrics": True,

    # ---- Target: residual to flat ----
    "use_flat_residual": True,
    "flat_file": "deeponet_aux/flat_ground.h5",

    # ---- Trunk input channels ----
    "use_eta_trunk": True,
    "use_mask_trunk": True,
    "use_pflat_trunk": True,
    "pflat_feat_norm": "zscore",

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
    "debug_no_mask": False,
    "debug_constant_target": False,
    "debug_constant_re": 0.0,
    "debug_constant_im": 0.0,

    "debug_shapes": False,  # prints shapes once inside model.forward
}


# ------------------------------ Utils / I/O ------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def case_family(h5_path: Path) -> str:
    name = h5_path.name
    if "__" in name:
        suf = name.split("__", 1)[-1]
        fam = suf.split(".")[0]
        if fam:
            return fam
    try:
        with h5py.File(h5_path, "r") as h:
            if "meta" in h and "family" in h["meta"].attrs:
                v = h["meta"].attrs["family"]
                if isinstance(v, (bytes, bytearray)):
                    return v.decode("utf-8")
                return str(v)
    except Exception:
        pass
    return "t?"

def load_grids(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h:
        xg = np.array(h["x_grid"][:], dtype=np.float32)
        zg = np.array(h["z_grid"][:], dtype=np.float32)
    return xg, zg

def load_h_profile(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h:
        x = np.array(h["h_profile"]["x"][:], dtype=np.float32)
        hv = np.array(h["h_profile"]["h"][:], dtype=np.float32)
    return x, hv

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
    raise ValueError(f"Field shape {field.shape} incompatible with (H,W)=({H},{W}).")

def ensure_complex_zx(pre: np.ndarray, pim: np.ndarray, xg: np.ndarray, zg: np.ndarray):
    return ensure_field_zx(pre, xg, zg), ensure_field_zx(pim, xg, zg)

def hx_on_xgrid(x_grid: np.ndarray, x_prof: np.ndarray, h_prof: np.ndarray) -> np.ndarray:
    if x_prof.shape == x_grid.shape and np.allclose(x_prof, x_grid):
        return h_prof.astype(np.float32)
    return np.interp(x_grid, x_prof, h_prof).astype(np.float32)

def air_mask_2d(xg: np.ndarray, zg: np.ndarray, hx_xgrid: np.ndarray) -> np.ndarray:
    return (zg[:, None] >= hx_xgrid[None, :])

def write_list(p: Path, items: List[Path]):
    ensure_dir(p.parent)
    p.write_text("\n".join(str(x) for x in items))

def read_list(p: Path) -> List[Path]:
    return [Path(s) for s in p.read_text().strip().splitlines() if s.strip()]

def uniform_sensors_x(xmin: float, xmax: float, m: int) -> np.ndarray:
    return np.linspace(xmin, xmax, m, dtype=np.float32)

def _assert_same_grid(xg_a: np.ndarray, zg_a: np.ndarray, xg_b: np.ndarray, zg_b: np.ndarray, name_a: str, name_b: str):
    if xg_a.shape != xg_b.shape or zg_a.shape != zg_b.shape:
        raise ValueError(f"Grid shape mismatch: {name_a} vs {name_b}")
    if (not np.allclose(xg_a, xg_b)) or (not np.allclose(zg_a, zg_b)):
        raise ValueError(f"Grid values mismatch: {name_a} vs {name_b}")


# ------------------------------ Flat baseline ------------------------------
def _flat_baseline_path(data_root: Path, cfg: Dict[str, Any]) -> Path:
    rel = str(cfg.get("flat_file", "deeponet_aux/flat_ground.h5"))
    p = Path(rel)
    return p if p.is_absolute() else (data_root / p)

def load_flat_baseline(
    data_root: Path,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_path = _flat_baseline_path(data_root, cfg)
    if not flat_path.exists():
        raise FileNotFoundError(f"Flat baseline not found: {flat_path}")
    xg, zg = load_grids(flat_path)
    pre_raw, pim_raw = load_p_re_im(flat_path)
    pre, pim = ensure_complex_zx(pre_raw, pim_raw, xg, zg)
    return xg, zg, pre, pim


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
        write_list(iid_dir / "val.txt",   val)
        write_list(iid_dir / "test.txt",  test)
        print(f"[splits] Wrote IID: {len(train)}/{len(val)}/{len(test)}")
    else:
        print("[splits] IID already exists -> skip")


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
    sensors_x: np.ndarray,
    cfg: Dict[str, Any],
    *,
    data_root: Path,
    flat: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    masked_stats = bool(cfg.get("masked_stats", True))
    use_flat_residual = bool(cfg.get("use_flat_residual", True))

    if (use_flat_residual or bool(cfg.get("use_pflat_trunk", True))) and flat is None:
        flat = load_flat_baseline(data_root, cfg)

    H_scale = float(cfg.get("H_scale", 100.0))
    h_values: List[np.ndarray] = []
    rs_re = RunningStats()
    rs_im = RunningStats()

    for p in train_list:
        x_prof, h_prof = load_h_profile(p)
        h_s = np.interp(sensors_x, x_prof, h_prof) / H_scale
        h_values.append(h_s.astype(np.float32))

        xg, zg = load_grids(p)
        pre_raw, pim_raw = load_p_re_im(p)
        pre, pim = ensure_complex_zx(pre_raw, pim_raw, xg, zg)

        if use_flat_residual:
            assert flat is not None
            xg_f, zg_f, flat_re, flat_im = flat
            _assert_same_grid(xg, zg, xg_f, zg_f, str(p), "flat_ground")
            pre = pre - flat_re
            pim = pim - flat_im

        if masked_stats:
            hx = hx_on_xgrid(xg, x_prof, h_prof).astype(np.float32).reshape(-1)
            mask2d = air_mask_2d(xg, zg, hx)
            rs_re.update(pre[mask2d])
            rs_im.update(pim[mask2d])
        else:
            rs_re.update(pre.reshape(-1))
            rs_im.update(pim.reshape(-1))

    h_cat = np.concatenate(h_values, axis=0) if h_values else np.zeros((0,), dtype=np.float32)
    h_mean = float(h_cat.mean()) if h_cat.size else 0.0
    h_std  = float(h_cat.std() + 1e-8) if h_cat.size else 1.0

    p_re_mean, p_re_std = rs_re.finalize()
    p_im_mean, p_im_std = rs_im.finalize()

    stats: Dict[str, Any] = {
        "h_mean": h_mean,
        "h_std":  h_std,
        "p_re_mean": p_re_mean,
        "p_re_std":  p_re_std,
        "p_im_mean": p_im_mean,
        "p_im_std":  p_im_std,
        "H_scale": float(H_scale),
        "masked_stats": bool(masked_stats),
        "use_flat_residual": bool(use_flat_residual),
        "flat_file": str(cfg.get("flat_file", "")) if use_flat_residual else "",
    }

    if bool(cfg.get("use_pflat_trunk", True)):
        assert flat is not None
        _, _, flat_re, flat_im = flat
        stats.update({
            "flat_re_mean": float(flat_re.mean()),
            "flat_re_std":  float(flat_re.std() + 1e-8),
            "flat_im_mean": float(flat_im.mean()),
            "flat_im_std":  float(flat_im.std() + 1e-8),
            "use_pflat_trunk": True,
            "pflat_feat_norm": str(cfg.get("pflat_feat_norm", "zscore")),
        })

    return stats


# ------------------------------ Dataset (grid) ------------------------------
class TrainDatasetGridResidual(Dataset):
    """
    Per-case return:
      branch_input: (m,)
      hx_xgrid:     (W,)  (forced 1D)
      y:            (2,H,W) normalized target (?p_re, ?p_im)
      mask:         (H,W) air mask
    """
    def __init__(
        self,
        paths: List[Path],
        sensors_x: np.ndarray,
        stats: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        data_root: Path,
        flat: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        self.paths = list(paths)
        self.sensors_x = sensors_x.astype(np.float32)
        self.stats = stats
        self.cfg = cfg
        self.data_root = data_root

        self.use_flat_residual = bool(cfg.get("use_flat_residual", True))
        self.flat = flat
        if (self.use_flat_residual or bool(cfg.get("use_pflat_trunk", True))) and self.flat is None:
            self.flat = load_flat_baseline(data_root, cfg)

        xg0, zg0 = load_grids(self.paths[0])
        self.xg = xg0
        self.zg = zg0

        if self.flat is not None:
            xg_f, zg_f, _, _ = self.flat
            _assert_same_grid(self.xg, self.zg, xg_f, zg_f, str(self.paths[0]), "flat_ground")

        # precompute branch inputs
        H_scale = float(stats.get("H_scale", cfg.get("H_scale", 100.0)))
        bis = []
        for p in self.paths:
            x_prof, h_prof = load_h_profile(p)
            h_s = np.interp(self.sensors_x, x_prof, h_prof) / H_scale
            h_s = (h_s - float(stats["h_mean"])) / float(stats["h_std"])
            bis.append(h_s.astype(np.float32))
        self.branch_inputs = np.stack(bis, axis=0)  # (N,m)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        xg, zg = load_grids(p)

        pre_raw, pim_raw = load_p_re_im(p)
        pre, pim = ensure_complex_zx(pre_raw, pim_raw, xg, zg)

        x_prof, h_prof = load_h_profile(p)
        hx = hx_on_xgrid(xg, x_prof, h_prof).astype(np.float32)
        hx = np.asarray(hx, dtype=np.float32).reshape(-1)  # FORCE (W,)
        mask2d = air_mask_2d(xg, zg, hx).astype(np.float32)

        if self.use_flat_residual:
            assert self.flat is not None
            _, _, flat_re, flat_im = self.flat
            pre = pre - flat_re
            pim = pim - flat_im

        y_re = (pre - float(self.stats["p_re_mean"])) / float(self.stats["p_re_std"])
        y_im = (pim - float(self.stats["p_im_mean"])) / float(self.stats["p_im_std"])

        if bool(self.cfg.get("debug_constant_target", False)):
            y_re[:] = float(self.cfg.get("debug_constant_re", 0.0))
            y_im[:] = float(self.cfg.get("debug_constant_im", 0.0))

        y = np.stack([y_re.astype(np.float32), y_im.astype(np.float32)], axis=0)  # (2,H,W)
        b = self.branch_inputs[idx]
        return {"branch_input": b, "hx_xgrid": hx, "y": y, "mask": mask2d, "path": str(p)}

def collate_cases_grid(batch: List[Dict[str, Any]]):
    branch_inputs = np.stack([b["branch_input"] for b in batch], axis=0).astype(np.float32)  # (B,m)

    hx_xgrid = np.stack([b["hx_xgrid"] for b in batch], axis=0).astype(np.float32)
    hx_xgrid = hx_xgrid.reshape(hx_xgrid.shape[0], -1)  # FORCE (B,W)

    y    = np.stack([b["y"] for b in batch], axis=0).astype(np.float32)     # (B,2,H,W)
    mask = np.stack([b["mask"] for b in batch], axis=0).astype(np.float32)  # (B,H,W)
    paths = [b["path"] for b in batch]

    return {
        "branch_inputs": torch.from_numpy(branch_inputs),
        "hx_xgrid":      torch.from_numpy(hx_xgrid),
        "y":             torch.from_numpy(y),
        "mask":          torch.from_numpy(mask),
        "paths":         paths,
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
    """
    FNO spectral conv with *safe AMP*:
      - we force FFT computations to FP32 (autocast disabled in this block)
      - avoids cuFFT fp16 power-of-two restrictions
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # FFT must be float32 to avoid ComplexHalf + cuFFT fp16 size restrictions
        if x.is_cuda:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                x_f32 = x.float()
                x_ft = torch.fft.rfft2(x_f32, norm="ortho")  # complex64

                out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)

                m1 = min(self.modes1, H)
                m2 = min(self.modes2, W // 2 + 1)

                out_ft[:, :, :m1, :m2]   = compl_mul2d(x_ft[:, :, :m1, :m2],   self.weights1[:, :, :m1, :m2])
                out_ft[:, :, -m1:, :m2]  = compl_mul2d(x_ft[:, :, -m1:, :m2],  self.weights2[:, :, :m1, :m2])

                y = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")  # float32
        else:
            # CPU: keep it simple, still float32 FFT
            x_f32 = x.float()
            x_ft = torch.fft.rfft2(x_f32, norm="ortho")
            out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)
            m1 = min(self.modes1, H)
            m2 = min(self.modes2, W // 2 + 1)
            out_ft[:, :, :m1, :m2]   = compl_mul2d(x_ft[:, :, :m1, :m2],   self.weights1[:, :, :m1, :m2])
            out_ft[:, :, -m1:, :m2]  = compl_mul2d(x_ft[:, :, -m1:, :m2],  self.weights2[:, :, :m1, :m2])
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
        self.spectral = nn.ModuleList([SpectralConv2d(int(width), int(width), int(modes_z), int(modes_x)) for _ in range(int(n_layers))])
        self.pointwise = nn.ModuleList([nn.Conv2d(int(width), int(width), kernel_size=1) for _ in range(int(n_layers))])
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
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

        x = self.lift(x)
        for k in range(len(self.spectral)):
            x = self.act(self.spectral[k](x) + self.pointwise[k](x))
        x = self.proj(x)

        if pad_h > 0 or pad_w > 0:
            x = x[..., :H, :W]
        return x

class DeepONetGridTrunkComplex(nn.Module):
    """
    forward(branch_inputs, hx_xgrid) -> (B,2,H,W) normalized ?p
    """
    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        xg: np.ndarray,
        zg: np.ndarray,
        stats: Dict[str, Any],
        flat: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        super().__init__()
        self.cfg = dict(cfg)

        self.use_eta   = bool(cfg.get("use_eta_trunk", True))
        self.use_mask  = bool(cfg.get("use_mask_trunk", True))
        self.use_pflat = bool(cfg.get("use_pflat_trunk", True))

        xg_t = torch.tensor(xg.astype(np.float32))
        zg_t = torch.tensor(zg.astype(np.float32))
        self.register_buffer("xg_buf", xg_t)
        self.register_buffer("zg_buf", zg_t)

        x_min, x_max = float(xg.min()), float(xg.max())
        z_min, z_max = float(zg.min()), float(zg.max())
        self.z_range = float(z_max - z_min + 1e-8)

        x_norm_1d = 2.0 * (xg_t - x_min) / (x_max - x_min + 1e-8) - 1.0
        z_norm_1d = 2.0 * (zg_t - z_min) / (z_max - z_min + 1e-8) - 1.0

        H = len(zg)
        W = len(xg)
        Xn = x_norm_1d[None, None, None, :].repeat(1, 1, H, 1)  # (1,1,H,W)
        Zn = z_norm_1d[None, None, :, None].repeat(1, 1, 1, W)  # (1,1,H,W)
        self.register_buffer("Xn", Xn)
        self.register_buffer("Zn", Zn)

        if self.use_pflat:
            if flat is None:
                raise ValueError("flat baseline must be provided when use_pflat_trunk=True")
            _, _, flat_re, flat_im = flat
            fr = torch.tensor(flat_re.astype(np.float32))[None, None, :, :]
            fi = torch.tensor(flat_im.astype(np.float32))[None, None, :, :]
            if str(cfg.get("pflat_feat_norm", "zscore")) == "zscore":
                fr = (fr - float(stats["flat_re_mean"])) / float(stats["flat_re_std"])
                fi = (fi - float(stats["flat_im_mean"])) / float(stats["flat_im_std"])
            self.register_buffer("flat_re_norm", fr)
            self.register_buffer("flat_im_norm", fi)
        else:
            self.register_buffer("flat_re_norm", torch.zeros(1, 1, H, W, dtype=torch.float32))
            self.register_buffer("flat_im_norm", torch.zeros(1, 1, H, W, dtype=torch.float32))

        m = int(cfg["branch_m"])
        L = int(cfg["latent"])
        self.L = L
        out_dim_lat = 2 * L

        self.branch = MLP(m, int(cfg["branch_width"]), int(cfg["branch_depth"]), out_dim_lat, act="relu")

        in_ch = 2  # x_norm, z_norm
        if self.use_eta:
            in_ch += 1
        if self.use_mask:
            in_ch += 1
        if self.use_pflat:
            in_ch += 2

        self.trunk = FNOGridTrunk(
            in_channels=in_ch,
            out_channels=out_dim_lat,
            width=int(cfg["trunk_width"]),
            n_layers=int(cfg["trunk_layers"]),
            modes_z=int(cfg["modes_z"]),
            modes_x=int(cfg["modes_x"]),
            padding_frac=float(cfg.get("padding_frac", 0.0)),
        )

        self._printed_shapes = False

    def forward(self, branch_inputs: torch.Tensor, hx_xgrid: torch.Tensor) -> torch.Tensor:
        """
        branch_inputs: (B, m)
        hx_xgrid:      (B, W) (or with extra singleton dims; sanitized)
        returns:       (B, 2, H, W) normalized ?p prediction
        """
        B = branch_inputs.shape[0]
        H = self.Zn.shape[2]
        W = self.Xn.shape[3]

        branch_inputs = branch_inputs.to(device=self.Xn.device)
        hx_xgrid = hx_xgrid.to(device=self.Xn.device, dtype=self.Xn.dtype)

        # sanitize hx_xgrid -> (B,W)
        if hx_xgrid.ndim == 1:
            hx_xgrid = hx_xgrid[None, :]
        while hx_xgrid.ndim > 2:
            if hx_xgrid.shape[1] == 1:
                hx_xgrid = hx_xgrid[:, 0, ...]
                continue
            if hx_xgrid.shape[-2] == 1:
                hx_xgrid = hx_xgrid.squeeze(-2)
                continue
            break
        if hx_xgrid.ndim != 2:
            raise ValueError(f"hx_xgrid must be 2D (B,W) but got {tuple(hx_xgrid.shape)}")
        if hx_xgrid.shape[0] == 1 and B > 1:
            hx_xgrid = hx_xgrid.expand(B, -1)
        if hx_xgrid.shape[0] != B:
            raise ValueError(f"hx_xgrid batch mismatch: got {hx_xgrid.shape[0]} expected {B}")
        if hx_xgrid.shape[1] != W:
            raise ValueError(f"hx_xgrid width mismatch: got {hx_xgrid.shape[1]} expected {W}")

        # branch -> (B,2,L)
        bvec = self.branch(branch_inputs).view(B, 2, self.L)

        feats: List[torch.Tensor] = []
        feats.append(self.Xn.repeat(B, 1, 1, 1))
        feats.append(self.Zn.repeat(B, 1, 1, 1))

        # hx: (B,1,1,W)
        hx = hx_xgrid[:, None, None, :]

        # zg_4: (1,H,1,1) => eta_raw should become (B,H,1,W)
        zg_4 = self.zg_buf.view(1, H, 1, 1).to(self.Xn.dtype)
        eta_raw = zg_4 - hx  # (B,H,1,W)

        # FORCE eta to (B,H,W) robustly
        if eta_raw.ndim == 4 and eta_raw.shape[2] == 1:
            eta = eta_raw.squeeze(2)          # (B,H,W)
        elif eta_raw.ndim == 4 and eta_raw.shape[1] == 1:
            eta = eta_raw[:, 0, :, :]         # (B,H,W)
        elif eta_raw.ndim == 3:
            eta = eta_raw                     # (B,H,W)
        else:
            raise ValueError(f"Unexpected eta_raw shape: {tuple(eta_raw.shape)}")

        if self.use_eta:
            eta_norm = (2.0 * (eta / float(self.z_range)) - 1.0).unsqueeze(1)  # (B,1,H,W)
            feats.append(eta_norm)

        if self.use_mask:
            air = (eta >= 0.0).to(self.Xn.dtype).unsqueeze(1)  # (B,1,H,W)
            feats.append(air)

        if self.use_pflat:
            feats.append(self.flat_re_norm.repeat(B, 1, 1, 1))
            feats.append(self.flat_im_norm.repeat(B, 1, 1, 1))

        if bool(self.cfg.get("debug_shapes", False)) and (not self._printed_shapes):
            print("[debug] hx_xgrid", tuple(hx_xgrid.shape))
            for i, f in enumerate(feats):
                print(f"[debug] feat[{i}] shape={tuple(f.shape)}")
            self._printed_shapes = True

        trunk_in = torch.cat(feats, dim=1)                  # (B,C,H,W)
        tfeat = self.trunk(trunk_in).view(B, 2, self.L, H, W)
        out = torch.einsum("bcl,bclhw->bchw", bvec, tfeat)  # (B,2,H,W)
        return out


# ------------------------------ Loss ------------------------------
def masked_mse_grid(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[:, None, :, :].to(pred.dtype)
    se = (pred - y) ** 2
    se = se * m
    denom = (m.sum() * pred.shape[1]).clamp_min(1e-8)
    return se.sum() / denom


# ------------------------------ Train ------------------------------
def train_run(
    data_root: Path,
    split: str,
    run_tag: str,
    device: str | None = None,
    cfg: Dict[str, Any] = CFG,
):
    cfg = dict(cfg)

    # NOTE: we still set seeds here; if resuming with RNG restore, we overwrite them after loading ckpt
    set_all_seeds(int(cfg["seed"]))
    torch.backends.cudnn.benchmark = True

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    splits_dir = data_root / "splits"
    sp_dir = splits_dir / split
    train_list = read_list(sp_dir / "train.txt")
    val_list   = read_list(sp_dir / "val.txt")

    if bool(cfg.get("debug_overfit", False)):
        n_dbg = max(1, min(int(cfg.get("debug_n_cases", 8)), len(train_list)))
        train_list = train_list[:n_dbg]
        val_list = list(train_list)
        print(f"[debug] OVERFIT mode ON -> using {len(train_list)} train/val cases")
        if bool(cfg.get("debug_no_mask", False)):
            cfg["masked_loss"] = False

    aux_dir = data_root / "deeponet_aux"
    ensure_dir(aux_dir)

    sensors_path = aux_dir / f"sensors_x_m{cfg['branch_m']}.npy"
    stats_suffix = "resflat" if bool(cfg.get("use_flat_residual", True)) else "full"
    stats_path   = aux_dir / f"stats_gridtrunk_{stats_suffix}_m{cfg['branch_m']}_L{cfg['latent']}.json"

    # sensors
    if sensors_path.exists():
        sensors_x = np.load(sensors_path).astype(np.float32)
        print(f"[train] Loaded sensors_x from {sensors_path}")
    else:
        xg0, _ = load_grids(train_list[0])
        sensors_x = uniform_sensors_x(float(xg0.min()), float(xg0.max()), int(cfg["branch_m"]))
        np.save(sensors_path, sensors_x)
        print(f"[train] Wrote sensors_x to {sensors_path}")

    # flat
    flat = None
    if bool(cfg.get("use_flat_residual", True)) or bool(cfg.get("use_pflat_trunk", True)):
        flat = load_flat_baseline(data_root, cfg)
        xg_flat, zg_flat, _, _ = flat
        xg0, zg0 = load_grids(train_list[0])
        _assert_same_grid(xg0, zg0, xg_flat, zg_flat, str(train_list[0]), "flat_ground")
        print(f"[train] Using flat file {_flat_baseline_path(data_root, cfg)}")

    # stats
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        print(f"[train] Loaded stats from {stats_path}")
    else:
        print("[train] Stats not found -> computing from train set (one-time cost)...")
        stats = compute_stats_from_train(train_list, sensors_x, cfg, data_root=data_root, flat=flat)
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"[train] Wrote stats to {stats_path}")

    xg0, zg0 = load_grids(train_list[0])

    tr_ds = TrainDatasetGridResidual(train_list, sensors_x, stats, cfg, data_root=data_root, flat=flat)
    va_ds = TrainDatasetGridResidual(val_list,   sensors_x, stats, cfg, data_root=data_root, flat=flat)

    tr_ld = DataLoader(
        tr_ds,
        batch_size=int(cfg["batch_cases"]),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_cases_grid,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=str(device).startswith("cuda"),
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=int(cfg["batch_cases"]),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_cases_grid,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=str(device).startswith("cuda"),
    )

    model = DeepONetGridTrunkComplex(cfg, xg=xg0, zg=zg0, stats=stats, flat=flat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    # AMP policy:
    # - autocast allowed
    # - GradScaler disabled if model has complex params (it will crash otherwise)
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

            # NEW: RNG states for deterministic resume
            "rng_torch": torch.get_rng_state(),
            "rng_cuda":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
            "rng_py":    random.getstate(),

            # metadata
            "cfg": cfg,
            "stats": stats,
            "sensors_x": sensors_x,
            "run_tag": run_tag,
            "split": split,
            "x_grid": xg0,
            "z_grid": zg0,
        }, tmp)
        tmp.replace(path)

    # ------------------------------ Resume (NEW) ------------------------------
    if RESUME_TRAINING and resume_path.exists():
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)


            # sanity: warn if run_tag mismatch (doesn't block)
            if ckpt.get("run_tag", run_tag) != run_tag:
                print(f"[resume] WARNING: checkpoint run_tag={ckpt.get('run_tag')} != current run_tag={run_tag}")

            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])

            if sched is not None and ckpt.get("sched") is not None:
                sched.load_state_dict(ckpt["sched"])

            if scaler is not None and scaler.is_enabled() and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])

            # restore best/earlystop state
            best_val = float(ckpt.get("best_val", best_val))
            patience_left = int(ckpt.get("patience_left", patience_left))

            # restore RNG states if present
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

    print(f"[train] split={split} device={device_info_string(str(device))}")
    print(f"[train] Ntrain={len(train_list)} Nval={len(val_list)} batch_cases={cfg['batch_cases']} autocast={use_autocast} scaler={bool(scaler and scaler.is_enabled())}")
    print(f"[train] latent L={cfg['latent']} trunk_width={cfg['trunk_width']} layers={cfg['trunk_layers']} modes=({cfg['modes_z']},{cfg['modes_x']}) pad={cfg.get('padding_frac',0.0)}")
    print(f"[train] trunk channels: eta={cfg.get('use_eta_trunk',True)} mask={cfg.get('use_mask_trunk',True)} pflat={cfg.get('use_pflat_trunk',True)}")
    print(f"[train] use_flat_residual={cfg.get('use_flat_residual',True)} masked_loss={cfg.get('masked_loss',True)}")

    last_epoch_ran = None
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        last_epoch_ran = epoch
        t0 = time.time()

        # train
        model.train()
        tr_loss = 0.0
        nb = 0
        for batch in tr_ld:
            branch_inputs = batch["branch_inputs"].to(device)
            hx_xgrid      = batch["hx_xgrid"].to(device)
            y             = batch["y"].to(device)
            mask          = batch["mask"].to(device)

            opt.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.amp.autocast(device_type=device_type, enabled=True):
                    pred = model(branch_inputs, hx_xgrid)
                    loss = masked_mse_grid(pred, y, mask) if bool(cfg.get("masked_loss", True)) else torch.mean((pred - y) ** 2)

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            else:
                pred = model(branch_inputs, hx_xgrid)
                loss = masked_mse_grid(pred, y, mask) if bool(cfg.get("masked_loss", True)) else torch.mean((pred - y) ** 2)
                loss.backward()
                opt.step()

            tr_loss += float(loss.detach().cpu())
            nb += 1
        tr_loss /= max(1, nb)

        # val
        model.eval()
        va_loss = 0.0
        nb2 = 0
        with torch.no_grad():
            for batch in va_ld:
                branch_inputs = batch["branch_inputs"].to(device)
                hx_xgrid      = batch["hx_xgrid"].to(device)
                y             = batch["y"].to(device)
                mask          = batch["mask"].to(device)

                if use_autocast:
                    with torch.amp.autocast(device_type=device_type, enabled=True):
                        pred = model(branch_inputs, hx_xgrid)
                        loss = masked_mse_grid(pred, y, mask) if bool(cfg.get("masked_loss", True)) else torch.mean((pred - y) ** 2)
                else:
                    pred = model(branch_inputs, hx_xgrid)
                    loss = masked_mse_grid(pred, y, mask) if bool(cfg.get("masked_loss", True)) else torch.mean((pred - y) ** 2)

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


# ------------------------------ Autorun ------------------------------
def run_everything():
    data_root = Path(DATA_ROOT)

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
        train_run(data_root, split="iid", run_tag=tag, device=device, cfg=CFG)

if __name__ == "__main__":
    run_everything()
