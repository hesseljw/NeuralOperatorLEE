# fno_terrain_complex_eta_mask_coords_phys.py
# Physics-informed FNO for terrain → complex pressure (or residual Δp).
#
# Key idea:
# - Train on normalized targets for stable optimization
# - Compute physics residuals on *physical pressure* p (de-normalized) to avoid affine-normalization biasing Helmholtz.
#
# Inputs  (4,H,W): [eta_norm, air_mask, x_norm, z_norm]
# Targets (2,H,W): [p_re, p_im] if use_flat_residual=False
#                : [Δp_re, Δp_im] where Δp = p_terrain - p_flat if use_flat_residual=True
# Reconstruction (if residual): p_pred = Δp_pred + p_flat

from __future__ import annotations
import json, time, random, math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ------------------------------ User toggles ------------------------------
DATA_ROOT = Path("data") / "terrain_sobol_complex_n_1000"
DEVICE    = None   # "cuda" or "cpu" (if None, auto-pick)
RUN_TAG   = "fno_phys_iid"    # checkpoints/<RUN_TAG>.pt, checkpoints/<RUN_TAG>_last.pt

RUN_IID        = True
RUN_LOFO_ALL   = False
RUN_RANGE_OOD  = False

CFG: Dict[str, Any] = {
    "seed": 1234,

    # ---- Training ----
    "epochs": 3000,
    "lr": 1e-5,                # good for transfer-learning finetune
    "weight_decay": 1e-5,
    "batch_size": 1,
    "num_workers": 0,
    "amp": True,

    # ---- Early stopping ----
    "early_stop": False,
    "patience": 40,
    "es_delta": 1e-4,

    # ---- Save ----
    "save_every": 1,

    # ---- Masking ----
    "masked_loss": True,       # supervise only in air (recommended)
    "masked_stats": True,      # compute normalization stats only in air (recommended)
    "masked_metrics": True,

    # ---- Learn Δp vs p ----
    "use_flat_residual": False,     # set True to learn Δp = p_terrain - p_flat
    "flat_file": "deeponet_aux/flat_ground.h5",

    # ---- Optional crop ----
    "crop_to_safe_x_window": False,
    "safe_x_window": (-450.0, 450.0),

    # ---- FNO ----
    "in_channels": 4,
    "out_channels": 2,
    "width": 64,
    "n_layers": 4,
    "modes_z": 24,
    "modes_x": 64,
    "padding_frac": 0.05,

    # ---- Scheduler ----
    "use_lr_scheduler": True,
    "sched_factor": 0.75,
    "sched_patience": 10,
    "sched_threshold": 1e-5,
    "sched_min_lr": 1e-6,
    "sched_cooldown": 0,
    "sched_metric": "val_data",  # "val_data" or "val_total"

    # ---- Physics losses ----
    "use_bc_loss": True,
    "use_pde_loss": True,

    # small weights; tune upward slowly
    "lambda_bc": 1e-5,
    "lambda_pde": 1e-7,

    # ramps / start
    "bc_ramp_epochs": 200,
    "pde_start_epoch": 300,
    "pde_ramp_epochs": 700,

    # PDE constants (match generator)
    "f0_hz": 25.0,
    "c0_ms": 340.0,

    # PDE masking
    "src_x_m": 0.0,
    "src_z_m": 92.0,
    "src_radius_cells": 10,
    "pde_edge_band_cells": 10,
    "pde_exclude_boundary_band": 2,
    "pde_every_n_steps": 1,

    # Range-OOD split param
    "extreme_pct": 15.0,

    # Transfer learning (optional):
    # - if empty: no init unless auto-resume finds _last / best checkpoint
    "init_ckpt": "checkpoints/fno_eta_mask_coords_learn_p_iid.pt",  # e.g. "checkpoints/fno_data_only_best.pt"
}


# ------------------------------ Utils ------------------------------
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
    raise ValueError(f"Field shape {field.shape} incompatible with (H,W)=({H},{W})")

def ensure_complex_zx(pre: np.ndarray, pim: np.ndarray, xg: np.ndarray, zg: np.ndarray):
    return ensure_field_zx(pre, xg, zg), ensure_field_zx(pim, xg, zg)

def x_crop_indices(xg: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return np.where((xg >= xmin) & (xg <= xmax))[0]

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

def load_manifest(data_root: Path) -> Dict[str, Any]:
    man = data_root / "manifest.json"
    if man.exists():
        return json.loads(man.read_text())
    return {}

# ------------------------------ Splits ------------------------------
def make_splits(data_root: Path, cfg=CFG):
    cases = find_cases(data_root)
    assert cases, f"No cases under {data_root}/cases"

    by_family: Dict[str, List[Path]] = {}
    for p in cases:
        by_family.setdefault(case_family(p), []).append(p)

    rng = np.random.default_rng(int(cfg["seed"]))

    # IID
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

    # LOFO
    lofo_dir = data_root / "splits" / "lofo"
    for fam in sorted(by_family):
        fam_dir = lofo_dir / fam
        if (fam_dir / "train.txt").exists():
            continue
        test = list(by_family[fam])
        pool = [p for f, lst in by_family.items() if f != fam for p in lst]
        rng.shuffle(pool)
        nva = max(1, int(0.15 * len(pool)))
        val = pool[:nva]
        train = pool[nva:]
        write_list(fam_dir / "train.txt", train)
        write_list(fam_dir / "val.txt",   val)
        write_list(fam_dir / "test.txt",  test)
        print(f"[splits] Wrote LOFO/{fam}: {len(train)}/{len(val)}/{len(test)}")

    # Range-OOD
    rdir = data_root / "splits" / "range_ood"
    if not (rdir / "train.txt").exists():
        heights, slopes = [], []
        for p in cases:
            x, h = load_h_profile(p)
            heights.append(float(np.max(h)))
            dh = np.gradient(h, x)
            slopes.append(float(np.max(np.abs(dh))))
        heights = np.asarray(heights, np.float32)
        slopes  = np.asarray(slopes,  np.float32)
        h0 = (heights - heights.min()) / (np.ptp(heights) + 1e-8)
        s0 = (slopes  - slopes.min())  / (np.ptp(slopes)  + 1e-8)
        sc = np.maximum(h0, s0)
        order = np.argsort(sc)[::-1]
        cases_arr = np.array(cases)

        extreme_pct = float(cfg.get("extreme_pct", 15.0))
        extreme_k = max(1, int(len(cases) * (extreme_pct / 100.0)))
        test = list(cases_arr[order[:extreme_k]])
        rest = list(cases_arr[order[extreme_k:]])
        rng.shuffle(rest)
        nva = max(1, int(0.15 * len(rest)))
        val = rest[:nva]
        train = rest[nva:]
        write_list(rdir / "train.txt", train)
        write_list(rdir / "val.txt",   val)
        write_list(rdir / "test.txt",  test)
        print(f"[splits] Wrote Range-OOD: {len(train)}/{len(val)}/{len(test)}")
    else:
        print("[splits] Range-OOD already exists -> skip")


# ------------------------------ Flat baseline ------------------------------
def _flat_baseline_path(data_root: Path, cfg: Dict[str, Any]) -> Path:
    rel = str(cfg.get("flat_file", "deeponet_aux/flat_ground.h5"))
    p = Path(rel)
    return p if p.is_absolute() else (data_root / p)

def load_flat_baseline(data_root: Path, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_path = _flat_baseline_path(data_root, cfg)
    if not flat_path.exists():
        raise FileNotFoundError(f"Flat baseline not found: {flat_path}")
    xg, zg = load_grids(flat_path)
    pre_raw, pim_raw = load_p_re_im(flat_path)
    pre, pim = ensure_complex_zx(pre_raw, pim_raw, xg, zg)
    return xg, zg, pre, pim

def _assert_same_grid(xg_a: np.ndarray, zg_a: np.ndarray, xg_b: np.ndarray, zg_b: np.ndarray, name_a: str, name_b: str):
    if xg_a.shape != xg_b.shape or zg_a.shape != zg_b.shape:
        raise ValueError(f"Grid shape mismatch {name_a} vs {name_b}.")
    if not np.allclose(xg_a, xg_b) or not np.allclose(zg_a, zg_b):
        raise ValueError(f"Grid values mismatch {name_a} vs {name_b}.")


# ------------------------------ Stats ------------------------------
def compute_stats_from_train(
    train_list: List[Path],
    cfg: Dict[str, Any],
    *,
    data_root: Path,
    flat: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    use_flat = bool(cfg.get("use_flat_residual", False))
    masked_stats = bool(cfg.get("masked_stats", True))
    if use_flat and flat is None:
        flat = load_flat_baseline(data_root, cfg)

    re_vals: List[np.ndarray] = []
    im_vals: List[np.ndarray] = []

    for p in train_list:
        xg, zg = load_grids(p)
        x_prof, h_prof = load_h_profile(p)
        hx = hx_on_xgrid(xg, x_prof, h_prof)
        pre_raw, pim_raw = load_p_re_im(p)
        pre, pim = ensure_complex_zx(pre_raw, pim_raw, xg, zg)

        idx = None
        if cfg.get("crop_to_safe_x_window", False):
            xmin, xmax = cfg.get("safe_x_window", (-450.0, 450.0))
            idx = x_crop_indices(xg, float(xmin), float(xmax))
            if idx.size >= 16:
                xg = xg[idx]
                pre = pre[:, idx]
                pim = pim[:, idx]
                hx  = hx[idx]

        if use_flat:
            assert flat is not None
            xgf, zgf, flat_re, flat_im = flat
            if idx is not None and idx.size >= 16:
                xgf2 = xgf[idx]
                flat_re2 = flat_re[:, idx]
                flat_im2 = flat_im[:, idx]
                _assert_same_grid(xg, zg, xgf2, zgf, str(p), "flat_ground(cropped)")
                dpre = pre - flat_re2
                dpim = pim - flat_im2
            else:
                _assert_same_grid(xg, zg, xgf, zgf, str(p), "flat_ground")
                dpre = pre - flat_re
                dpim = pim - flat_im
        else:
            dpre, dpim = pre, pim

        if masked_stats:
            m = air_mask_2d(xg, zg, hx)
            re_vals.append(dpre[m].astype(np.float32))
            im_vals.append(dpim[m].astype(np.float32))
        else:
            re_vals.append(dpre.reshape(-1).astype(np.float32))
            im_vals.append(dpim.reshape(-1).astype(np.float32))

    re_cat = np.concatenate(re_vals) if re_vals else np.zeros((0,), np.float32)
    im_cat = np.concatenate(im_vals) if im_vals else np.zeros((0,), np.float32)

    return {
        "p_re_mean": float(re_cat.mean()) if re_cat.size else 0.0,
        "p_re_std":  float(re_cat.std() + 1e-8) if re_cat.size else 1.0,
        "p_im_mean": float(im_cat.mean()) if im_cat.size else 0.0,
        "p_im_std":  float(im_cat.std() + 1e-8) if im_cat.size else 1.0,
        "masked_stats": masked_stats,
        "use_flat_residual": use_flat,
        "flat_file": str(cfg.get("flat_file", "")) if use_flat else "",
    }


# ------------------------------ Dataset ------------------------------
class TerrainDataset(Dataset):
    """
    Returns dict:
      X:     (4,H,W) float32
      Y:     (2,H,W) float32 (normalized target)
      mask:  (H,W)   float32 air mask
      flatY: (2,H,W) float32 normalized flat p (only meaningful if use_flat_residual=True)
      dx,dz: float32 physical grid spacing
      src_ix, src_iz: int indices
    """
    def __init__(
        self,
        paths: List[Path],
        stats: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        data_root: Path,
        flat: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        self.paths = paths
        self.stats = stats
        self.cfg = cfg
        self.data_root = data_root
        self.use_flat = bool(cfg.get("use_flat_residual", False))
        self.flat = flat
        if self.use_flat and self.flat is None:
            self.flat = load_flat_baseline(data_root, cfg)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        xg, zg = load_grids(p)
        x_prof, h_prof = load_h_profile(p)
        hx = hx_on_xgrid(xg, x_prof, h_prof)
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
                hx  = hx[idx_x]

        H, W = len(zg), len(xg)

        # target: p or Δp
        if self.use_flat:
            assert self.flat is not None
            xgf, zgf, flat_re, flat_im = self.flat
            if idx_x is not None and idx_x.size >= 16:
                xgf2 = xgf[idx_x]
                flat_re2 = flat_re[:, idx_x]
                flat_im2 = flat_im[:, idx_x]
                _assert_same_grid(xg, zg, xgf2, zgf, str(p), "flat_ground(cropped)")
                tgt_re = pre - flat_re2
                tgt_im = pim - flat_im2
                flat_re_use, flat_im_use = flat_re2, flat_im2
            else:
                _assert_same_grid(xg, zg, xgf, zgf, str(p), "flat_ground")
                tgt_re = pre - flat_re
                tgt_im = pim - flat_im
                flat_re_use, flat_im_use = flat_re, flat_im
        else:
            tgt_re, tgt_im = pre, pim
            flat_re_use = np.zeros((H, W), np.float32)
            flat_im_use = np.zeros((H, W), np.float32)

        # normalize target
        y_re = (tgt_re - self.stats["p_re_mean"]) / self.stats["p_re_std"]
        y_im = (tgt_im - self.stats["p_im_mean"]) / self.stats["p_im_std"]
        Y = np.stack([y_re, y_im], axis=0).astype(np.float32)

        # geometry channels
        Z = zg[:, None].astype(np.float32)
        HX = hx[None, :].astype(np.float32)
        eta = Z - HX
        mask = (eta >= 0.0).astype(np.float32)

        z_range = float(zg.max() - zg.min() + 1e-8)
        eta_norm = (eta / z_range).astype(np.float32)

        x_norm_1d = (2.0 * (xg - xg.min()) / (xg.max() - xg.min() + 1e-8) - 1.0).astype(np.float32)
        z_norm_1d = (2.0 * (zg - zg.min()) / (zg.max() - zg.min() + 1e-8) - 1.0).astype(np.float32)
        x_norm = np.broadcast_to(x_norm_1d[None, :], (H, W))
        z_norm = np.broadcast_to(z_norm_1d[:, None], (H, W))

        X = np.stack([eta_norm, mask, x_norm, z_norm], axis=0).astype(np.float32)

        dx = float(np.mean(np.diff(xg))) if len(xg) > 1 else 1.0
        dz = float(np.mean(np.diff(zg))) if len(zg) > 1 else 1.0

        # flat baseline in normalized space (so we can reconstruct p_norm if desired)
        if self.use_flat:
            flat_y_re = (flat_re_use - self.stats["p_re_mean"]) / self.stats["p_re_std"]
            flat_y_im = (flat_im_use - self.stats["p_im_mean"]) / self.stats["p_im_std"]
            flatY = np.stack([flat_y_re, flat_y_im], axis=0).astype(np.float32)
        else:
            flatY = np.zeros((2, H, W), np.float32)

        src_x = float(self.cfg.get("src_x_m", 0.0))
        src_z = float(self.cfg.get("src_z_m", 92.0))
        src_ix = int(np.argmin(np.abs(xg - src_x))) if len(xg) else 0
        src_iz = int(np.argmin(np.abs(zg - src_z))) if len(zg) else 0

        return {
            "X": X, "Y": Y, "mask": mask, "flatY": flatY,
            "dx": dx, "dz": dz, "src_ix": src_ix, "src_iz": src_iz,
            "path": str(p),
        }

def collate(batch: List[Dict[str, Any]]):
    X = torch.from_numpy(np.stack([b["X"] for b in batch], 0))
    Y = torch.from_numpy(np.stack([b["Y"] for b in batch], 0))
    M = torch.from_numpy(np.stack([b["mask"] for b in batch], 0))
    flatY = torch.from_numpy(np.stack([b["flatY"] for b in batch], 0))
    dx = torch.tensor([b["dx"] for b in batch], dtype=torch.float32)
    dz = torch.tensor([b["dz"] for b in batch], dtype=torch.float32)
    src_ix = torch.tensor([b["src_ix"] for b in batch], dtype=torch.int64)
    src_iz = torch.tensor([b["src_iz"] for b in batch], dtype=torch.int64)
    paths = [b["path"] for b in batch]
    return {"X": X, "Y": Y, "mask": M, "flatY": flatY, "dx": dx, "dz": dz, "src_ix": src_ix, "src_iz": src_iz, "paths": paths}


# ------------------------------ FNO ------------------------------
class SpectralConv2d(nn.Module):
    """2D spectral conv with truncated Fourier modes; FFT runs float32 to avoid cuFFT fp16 restrictions."""
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
        x_fft = x.float() if orig_dtype in (torch.float16, torch.bfloat16) else x
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
    def __init__(self, in_channels: int, out_channels: int, width: int, n_layers: int, modes_z: int, modes_x: int, padding_frac: float):
        super().__init__()
        self.padding_frac = float(padding_frac)
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes_z, modes_x) for _ in range(n_layers)])
        self.w1x1 = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.InstanceNorm2d(width, affine=True) for _ in range(n_layers)])
        self.proj1 = nn.Conv2d(width, width, 1)
        self.proj2 = nn.Conv2d(width, out_channels, 1)

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


# ------------------------------ Losses ------------------------------
def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[:, None].to(pred.dtype)
    se = (pred - y) ** 2 * m
    denom = (m.sum() * pred.shape[1]).clamp_min(1e-8)
    return se.sum() / denom

def denorm_to_phys(pred_norm: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    p = pred_norm.to(torch.float32)
    out = torch.empty_like(p)
    out[:, 0] = p[:, 0] * float(stats["p_re_std"]) + float(stats["p_re_mean"])
    out[:, 1] = p[:, 1] * float(stats["p_im_std"]) + float(stats["p_im_mean"])
    return out

def _central_diff_x(f: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    B, C, H, W = f.shape
    out = torch.zeros_like(f)
    if W < 3:
        return out
    dxv = dx.view(B, 1, 1, 1).to(f.dtype)
    out[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2.0 * dxv)
    return out

def _central_diff_z(f: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    B, C, H, W = f.shape
    out = torch.zeros_like(f)
    if H < 3:
        return out
    dzv = dz.view(B, 1, 1, 1).to(f.dtype)
    out[:, :, 1:-1, :] = (f[:, :, 2:, :] - f[:, :, :-2, :]) / (2.0 * dzv)
    return out

def _second_diff_x(f: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    B, C, H, W = f.shape
    out = torch.zeros_like(f)
    if W < 3:
        return out
    dxv = dx.view(B, 1, 1, 1).to(f.dtype)
    out[..., 1:-1] = (f[..., 2:] - 2.0 * f[..., 1:-1] + f[..., :-2]) / (dxv * dxv)
    return out

def _second_diff_z(f: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    B, C, H, W = f.shape
    out = torch.zeros_like(f)
    if H < 3:
        return out
    dzv = dz.view(B, 1, 1, 1).to(f.dtype)
    out[:, :, 1:-1, :] = (f[:, :, 2:, :] - 2.0 * f[:, :, 1:-1, :] + f[:, :, :-2, :]) / (dzv * dzv)
    return out

def _interior_mask_like(mask: torch.Tensor) -> torch.Tensor:
    B, H, W = mask.shape
    out = torch.zeros_like(mask)
    if H >= 3 and W >= 3:
        out[:, 1:-1, 1:-1] = 1.0
    return out

def _interior_band_mask_like(mask: torch.Tensor, band: int) -> torch.Tensor:
    if band <= 0:
        return torch.ones_like(mask)
    B, H, W = mask.shape
    out = torch.zeros_like(mask)
    if H > 2 * band and W > 2 * band:
        out[:, band:-band, band:-band] = 1.0
    return out

def _disk_mask_batch(B: int, H: int, W: int, cz: torch.Tensor, cx: torch.Tensor, r: int, device, dtype) -> torch.Tensor:
    if r <= 0:
        return torch.zeros((B, H, W), device=device, dtype=dtype)
    zz = torch.arange(H, device=device).view(1, H, 1)
    xx = torch.arange(W, device=device).view(1, 1, W)
    masks = []
    rr2 = float(r * r)
    for b in range(B):
        dz2 = (zz - int(cz[b].item())) ** 2
        dx2 = (xx - int(cx[b].item())) ** 2
        m = (dz2 + dx2 <= rr2).to(dtype)
        masks.append(m)
    return torch.cat(masks, dim=0)

def build_boundary_mask(air_mask: torch.Tensor) -> torch.Tensor:
    ground = 1.0 - air_mask
    g = ground.unsqueeze(1)
    neigh_ground = F.max_pool2d(g, kernel_size=3, stride=1, padding=1).squeeze(1)
    boundary = air_mask * (neigh_ground > 0.0).to(air_mask.dtype)
    boundary = boundary * _interior_mask_like(air_mask)
    return boundary

def dilate_mask(m: torch.Tensor, iters: int) -> torch.Tensor:
    if iters <= 0:
        return m
    x = m.unsqueeze(1)
    for _ in range(iters):
        x = (F.max_pool2d(x, kernel_size=3, stride=1, padding=1) > 0.0).to(m.dtype)
    return x.squeeze(1)

def rigid_bc_loss(p_phys: torch.Tensor, eta_norm: torch.Tensor, air_mask: torch.Tensor, dx: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    boundary = build_boundary_mask(air_mask)
    if boundary.sum().item() < 1:
        return p_phys.new_tensor(0.0)

    eta = eta_norm.unsqueeze(1).to(p_phys.dtype)
    deta_dx = _central_diff_x(eta, dx)[:, 0]
    deta_dz = _central_diff_z(eta, dz)[:, 0]
    nrm = torch.sqrt(deta_dx * deta_dx + deta_dz * deta_dz + 1e-12)
    nxh = deta_dx / nrm
    nzh = deta_dz / nrm

    p = p_phys.to(torch.float32)
    dp_dx = _central_diff_x(p, dx)
    dp_dz = _central_diff_z(p, dz)

    dn_re = nxh[:, None] * dp_dx[:, 0:1] + nzh[:, None] * dp_dz[:, 0:1]
    dn_im = nxh[:, None] * dp_dx[:, 1:2] + nzh[:, None] * dp_dz[:, 1:2]

    bc_se = (dn_re ** 2 + dn_im ** 2).squeeze(1)
    denom = boundary.sum().clamp_min(1.0)
    return (bc_se * boundary).sum() / denom

def helmholtz_pde_loss(
    p_phys: torch.Tensor,
    air_mask: torch.Tensor,
    dx: torch.Tensor,
    dz: torch.Tensor,
    k2: float,
    *,
    exclude_band: int,
    src_ix: torch.Tensor,
    src_iz: torch.Tensor,
    src_radius_cells: int,
    edge_band_cells: int,
) -> torch.Tensor:
    boundary = build_boundary_mask(air_mask)
    band = dilate_mask(boundary, exclude_band)
    interior = air_mask * (1.0 - band)
    interior = interior * _interior_mask_like(air_mask)
    if edge_band_cells > 0:
        interior = interior * _interior_band_mask_like(air_mask, int(edge_band_cells))
    if src_radius_cells > 0:
        B, H, W = air_mask.shape
        src = _disk_mask_batch(B, H, W, cz=src_iz, cx=src_ix, r=int(src_radius_cells),
                               device=air_mask.device, dtype=air_mask.dtype)
        interior = interior * (1.0 - src)

    if interior.sum().item() < 1:
        return p_phys.new_tensor(0.0)

    p = p_phys.to(torch.float32)
    lap = _second_diff_x(p, dx) + _second_diff_z(p, dz)
    R = lap + (float(k2) * p)
    R2 = (R[:, 0] ** 2 + R[:, 1] ** 2)
    denom = interior.sum().clamp_min(1.0)
    return (R2 * interior).sum() / denom


# ------------------------------ Train / Eval ------------------------------
def train_run(data_root: Path, split: str, run_tag: str, *, lofo_family: str | None = None, device: str | None = None, cfg=CFG):
    cfg = dict(cfg)
    set_all_seeds(int(cfg["seed"]))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    man = load_manifest(data_root)
    if cfg.get("crop_to_safe_x_window", False) and "safe_x_window" in man:
        cfg["safe_x_window"] = tuple(man["safe_x_window"])

    splits_dir = data_root / "splits"
    if split == "iid":
        sp_dir = splits_dir / "iid"
    elif split == "range_ood":
        sp_dir = splits_dir / "range_ood"
    elif split == "lofo":
        assert lofo_family is not None
        sp_dir = splits_dir / "lofo" / lofo_family
    else:
        raise ValueError("split must be iid, lofo, or range_ood")

    train_list = read_list(sp_dir / "train.txt")
    val_list   = read_list(sp_dir / "val.txt")

    flat = None
    if bool(cfg.get("use_flat_residual", False)):
        flat = load_flat_baseline(data_root, cfg)
        xgf, zgf, _, _ = flat
        xg0, zg0 = load_grids(train_list[0])
        _assert_same_grid(xg0, zg0, xgf, zgf, str(train_list[0]), "flat_ground")
        print(f"[train] Using Δp target + flat reconstruction (flat={_flat_baseline_path(data_root, cfg)})")
    else:
        print("[train] Learning p directly (no flat residual).")

    aux_dir = data_root / "fno_aux"
    ensure_dir(aux_dir)
    stats_suffix = "resflat" if bool(cfg.get("use_flat_residual", False)) else "full"
    stats_path = aux_dir / f"stats_complex_{stats_suffix}.json"

    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        ok = (bool(stats.get("use_flat_residual", False)) == bool(cfg.get("use_flat_residual", False))) and \
             (bool(stats.get("masked_stats", False)) == bool(cfg.get("masked_stats", True)))
        if not ok:
            stats = compute_stats_from_train(train_list, cfg, data_root=data_root, flat=flat)
            stats_path.write_text(json.dumps(stats, indent=2))
            print("[train] Stats changed -> recomputed.")
        else:
            print(f"[train] Loaded stats: {stats_path}")
    else:
        stats = compute_stats_from_train(train_list, cfg, data_root=data_root, flat=flat)
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"[train] Wrote stats: {stats_path}")

    tr_ds = TerrainDataset(train_list, stats, cfg, data_root=data_root, flat=flat)
    va_ds = TerrainDataset(val_list,   stats, cfg, data_root=data_root, flat=flat)

    tr_ld = DataLoader(tr_ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True,
                       num_workers=int(cfg.get("num_workers", 0)),
                       pin_memory=device.startswith("cuda"), collate_fn=collate)
    va_ld = DataLoader(va_ds, batch_size=int(cfg["batch_size"]), shuffle=False, drop_last=False,
                       num_workers=int(cfg.get("num_workers", 0)),
                       pin_memory=device.startswith("cuda"), collate_fn=collate)

    model = FNO2d(int(cfg["in_channels"]), int(cfg["out_channels"]),
                  int(cfg["width"]), int(cfg["n_layers"]),
                  int(cfg["modes_z"]), int(cfg["modes_x"]),
                  float(cfg.get("padding_frac", 0.0))).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    use_amp = bool(cfg.get("amp", False)) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    ckpt_dir = Path("checkpoints")
    ensure_dir(ckpt_dir)
    best_ckpt = ckpt_dir / f"{run_tag}.pt"
    last_ckpt = ckpt_dir / f"{run_tag}_last.pt"

    # resume if possible
    best_val = float("inf")
    start_epoch = 1
    resume_path = last_ckpt if last_ckpt.exists() else (best_ckpt if best_ckpt.exists() else None)
    if resume_path is not None:
        st = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if use_amp and scaler is not None and st.get("scaler") is not None:
            try:
                scaler.load_state_dict(st["scaler"])
            except Exception:
                pass
        best_val = float(st.get("best_val", best_val))
        start_epoch = int(st.get("epoch", 0)) + 1
        print(f"[train] Resumed from {resume_path} at epoch {start_epoch}")

    # optional init_ckpt if not resumed
    init_ckpt = str(cfg.get("init_ckpt", "")).strip()
    if resume_path is None and init_ckpt:
        ip = Path(init_ckpt)
        if ip.exists():
            st = torch.load(ip, map_location=device, weights_only=False)
            if isinstance(st, dict) and "model" in st:
                model.load_state_dict(st["model"], strict=True)
            elif isinstance(st, dict):
                model.load_state_dict(st, strict=True)
            print(f"[train] Initialized from {ip}")
        else:
            print(f"[train] WARNING init_ckpt not found: {ip}")

    sched = None
    if bool(cfg.get("use_lr_scheduler", True)):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min",
            factor=float(cfg.get("sched_factor", 0.75)),
            patience=int(cfg.get("sched_patience", 10)),
            threshold=float(cfg.get("sched_threshold", 1e-5)),
            min_lr=float(cfg.get("sched_min_lr", 1e-6)),
            cooldown=int(cfg.get("sched_cooldown", 0)),
        )

    def save_state(path: Path, epoch_i: int, best_val_local: float):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "epoch": epoch_i,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if (use_amp and scaler is not None) else None,
            "best_val": best_val_local,
            "cfg": cfg,
            "stats": stats,
            "run_tag": run_tag,
            "split": split,
            "lofo_family": lofo_family,
        }, tmp)
        tmp.replace(path)

    print(f"[train] split={split} lofo={lofo_family} device={device_info_string(device)} amp={use_amp}")
    print(f"[train] Ntrain={len(train_list)} Nval={len(val_list)}")
    print(f"[train] use_flat_residual={cfg.get('use_flat_residual')} masked_loss={cfg.get('masked_loss')} masked_stats={cfg.get('masked_stats')}")
    print(f"[train] λ_bc={cfg.get('lambda_bc')} λ_pde={cfg.get('lambda_pde')} pde_start={cfg.get('pde_start_epoch')}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    save_every = max(1, int(cfg.get("save_every", 1)))
    global_step = 0

    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        t0 = time.time()

        bc_ramp = max(1, int(cfg.get("bc_ramp_epochs", 1)))
        pde_ramp = max(1, int(cfg.get("pde_ramp_epochs", 1)))
        lam_bc = float(cfg.get("lambda_bc", 0.0)) * min(1.0, epoch / bc_ramp) if bool(cfg.get("use_bc_loss", True)) else 0.0
        lam_pde = 0.0
        if bool(cfg.get("use_pde_loss", True)) and epoch >= int(cfg.get("pde_start_epoch", 0)):
            lam_pde = float(cfg.get("lambda_pde", 0.0)) * min(1.0, (epoch - int(cfg.get("pde_start_epoch", 0)) + 1) / pde_ramp)

        model.train()
        tr_loss = tr_data = tr_bc = tr_pde = 0.0
        nb = 0

        for batch in tr_ld:
            X = batch["X"].to(device, non_blocking=True)
            Y = batch["Y"].to(device, non_blocking=True)
            M = batch["mask"].to(device, non_blocking=True)
            flatY = batch["flatY"].to(device, non_blocking=True)
            dx = batch["dx"].to(device, non_blocking=True)
            dz = batch["dz"].to(device, non_blocking=True)
            src_ix = batch["src_ix"].to(device, non_blocking=True)
            src_iz = batch["src_iz"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                assert scaler is not None
                with torch.amp.autocast(device_type=device_type, enabled=True):
                    pred = model(X)
                    data_loss = masked_mse(pred, Y, M) if bool(cfg.get("masked_loss", True)) else torch.mean((pred - Y) ** 2)

                    # physics on physical p
                    if bool(cfg.get("use_flat_residual", False)):
                        dp_phys = denorm_to_phys(pred, stats)
                        flat_phys = denorm_to_phys(flatY, stats)
                        p_phys = dp_phys + flat_phys
                    else:
                        p_phys = denorm_to_phys(pred, stats)

                    bc_loss = pred.new_tensor(0.0)
                    pde_loss = pred.new_tensor(0.0)

                    if bool(cfg.get("use_bc_loss", True)) and lam_bc > 0.0:
                        bc_loss = rigid_bc_loss(p_phys, X[:, 0], M, dx, dz)

                    if bool(cfg.get("use_pde_loss", True)) and lam_pde > 0.0:
                        every = max(1, int(cfg.get("pde_every_n_steps", 1)))
                        if (global_step % every) == 0:
                            f0 = float(cfg.get("f0_hz", 25.0))
                            c0 = float(cfg.get("c0_ms", 340.0))
                            k2 = (2.0 * math.pi * f0 / c0) ** 2
                            pde_loss = helmholtz_pde_loss(
                                p_phys, M, dx, dz, k2=k2,
                                exclude_band=int(cfg.get("pde_exclude_boundary_band", 2)),
                                src_ix=src_ix, src_iz=src_iz,
                                src_radius_cells=int(cfg.get("src_radius_cells", 0)),
                                edge_band_cells=int(cfg.get("pde_edge_band_cells", 0)),
                            )

                    loss = data_loss + lam_bc * bc_loss + lam_pde * pde_loss

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(X)
                data_loss = masked_mse(pred, Y, M) if bool(cfg.get("masked_loss", True)) else torch.mean((pred - Y) ** 2)

                if bool(cfg.get("use_flat_residual", False)):
                    dp_phys = denorm_to_phys(pred, stats)
                    flat_phys = denorm_to_phys(flatY, stats)
                    p_phys = dp_phys + flat_phys
                else:
                    p_phys = denorm_to_phys(pred, stats)

                bc_loss = pred.new_tensor(0.0)
                pde_loss = pred.new_tensor(0.0)
                if bool(cfg.get("use_bc_loss", True)) and lam_bc > 0.0:
                    bc_loss = rigid_bc_loss(p_phys, X[:, 0], M, dx, dz)
                if bool(cfg.get("use_pde_loss", True)) and lam_pde > 0.0:
                    every = max(1, int(cfg.get("pde_every_n_steps", 1)))
                    if (global_step % every) == 0:
                        f0 = float(cfg.get("f0_hz", 25.0))
                        c0 = float(cfg.get("c0_ms", 340.0))
                        k2 = (2.0 * math.pi * f0 / c0) ** 2
                        pde_loss = helmholtz_pde_loss(
                            p_phys, M, dx, dz, k2=k2,
                            exclude_band=int(cfg.get("pde_exclude_boundary_band", 2)),
                            src_ix=src_ix, src_iz=src_iz,
                            src_radius_cells=int(cfg.get("src_radius_cells", 0)),
                            edge_band_cells=int(cfg.get("pde_edge_band_cells", 0)),
                        )
                loss = data_loss + lam_bc * bc_loss + lam_pde * pde_loss
                loss.backward()
                opt.step()

            tr_loss += float(loss.detach().cpu())
            tr_data += float(data_loss.detach().cpu())
            tr_bc += float(bc_loss.detach().cpu())
            tr_pde += float(pde_loss.detach().cpu())
            nb += 1
            global_step += 1

        tr_loss /= max(1, nb)
        tr_data /= max(1, nb)
        tr_bc /= max(1, nb)
        tr_pde /= max(1, nb)

        # val
        model.eval()
        va_loss = va_data = va_bc = va_pde = 0.0
        nb2 = 0
        with torch.no_grad():
            for batch in va_ld:
                X = batch["X"].to(device, non_blocking=True)
                Y = batch["Y"].to(device, non_blocking=True)
                M = batch["mask"].to(device, non_blocking=True)
                flatY = batch["flatY"].to(device, non_blocking=True)
                dx = batch["dx"].to(device, non_blocking=True)
                dz = batch["dz"].to(device, non_blocking=True)
                src_ix = batch["src_ix"].to(device, non_blocking=True)
                src_iz = batch["src_iz"].to(device, non_blocking=True)

                pred = model(X)
                data_loss = masked_mse(pred, Y, M) if bool(cfg.get("masked_loss", True)) else torch.mean((pred - Y) ** 2)

                if bool(cfg.get("use_flat_residual", False)):
                    dp_phys = denorm_to_phys(pred, stats)
                    flat_phys = denorm_to_phys(flatY, stats)
                    p_phys = dp_phys + flat_phys
                else:
                    p_phys = denorm_to_phys(pred, stats)

                bc_loss = pred.new_tensor(0.0)
                pde_loss = pred.new_tensor(0.0)
                if bool(cfg.get("use_bc_loss", True)) and lam_bc > 0.0:
                    bc_loss = rigid_bc_loss(p_phys, X[:, 0], M, dx, dz)
                if bool(cfg.get("use_pde_loss", True)) and lam_pde > 0.0:
                    f0 = float(cfg.get("f0_hz", 25.0))
                    c0 = float(cfg.get("c0_ms", 340.0))
                    k2 = (2.0 * math.pi * f0 / c0) ** 2
                    pde_loss = helmholtz_pde_loss(
                        p_phys, M, dx, dz, k2=k2,
                        exclude_band=int(cfg.get("pde_exclude_boundary_band", 2)),
                        src_ix=src_ix, src_iz=src_iz,
                        src_radius_cells=int(cfg.get("src_radius_cells", 0)),
                        edge_band_cells=int(cfg.get("pde_edge_band_cells", 0)),
                    )
                loss = data_loss + lam_bc * bc_loss + lam_pde * pde_loss

                va_loss += float(loss.detach().cpu())
                va_data += float(data_loss.detach().cpu())
                va_bc += float(bc_loss.detach().cpu())
                va_pde += float(pde_loss.detach().cpu())
                nb2 += 1

        va_loss /= max(1, nb2)
        va_data /= max(1, nb2)
        va_bc /= max(1, nb2)
        va_pde /= max(1, nb2)

        if sched is not None:
            metric = str(cfg.get("sched_metric", "val_data")).lower().strip()
            sched.step(va_data if metric in ("val_data", "data", "va_data") else va_loss)

        lr_now = float(opt.param_groups[0]["lr"])
        dt = time.time() - t0
        print(
            f"[epoch {epoch:04d}] "
            f"train={tr_loss:.6f} (data={tr_data:.6f}, bc={tr_bc:.6f}, pde={tr_pde:.6f})  "
            f"val={va_loss:.6f} (data={va_data:.6f}, bc={va_bc:.6f}, pde={va_pde:.6f})  "
            f"λ_bc={lam_bc:.2e} λ_pde={lam_pde:.2e} lr={lr_now:.2e} ({dt:.1f}s)"
        )

        if va_loss < best_val - float(cfg.get("es_delta", 0.0)):
            best_val = va_loss
            save_state(best_ckpt, epoch, best_val)
            print(f"  → saved BEST to {best_ckpt}")

        if epoch % save_every == 0:
            save_state(last_ckpt, epoch, best_val)

    save_state(last_ckpt, int(cfg["epochs"]), best_val)
    print(f"[train] saved LAST to {last_ckpt}")
    print(f"[train] best val {best_val:.6f} (BEST={best_ckpt})")


# ------------------------------ Autorun ------------------------------
def run_everything():
    data_root = Path(DATA_ROOT)
    if not (data_root / "splits" / "iid" / "train.txt").exists():
        print("[autorun] Splits not found -> creating...")
        make_splits(data_root)
    else:
        print("[autorun] Splits present -> skip make_splits.")

    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device_info_string(device)}")

    if RUN_IID:
        train_run(data_root, split="iid", run_tag=RUN_TAG, device=device)

    if RUN_LOFO_ALL:
        fams = sorted({case_family(p) for p in find_cases(data_root)})
        for fam in fams:
            train_run(data_root, split="lofo", lofo_family=fam, run_tag=f"{RUN_TAG}_lofo_{fam}", device=device)

    if RUN_RANGE_OOD:
        train_run(data_root, split="range_ood", run_tag=f"{RUN_TAG}_range_ood", device=device)

if __name__ == "__main__":
    run_everything()
