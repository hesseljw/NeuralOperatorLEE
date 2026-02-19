"""train_fno_wind_terrain_eta_mask.py

FNO surrogate for joint WIND + TERRAIN cases.

Maps (U(z), h(x)) -> complex pressure field p(x,z) on a fixed ROI grid.
Training target is typically the *residual* to a flat-ground baseline:
    Δp = p(U,h) - p0
and we reconstruct:
    p_pred = Δp_pred + p0

Inputs (default 5 channels on the ROI grid):
  1) U_grid_norm : normalized U(z) broadcast over x
  2) eta_norm    : (z - h(x)) / z_range
  3) air_mask    : 1[z >= h(x)]
  4) x_norm
  5) z_norm
Optionally:
  + dU/dz channel (--use-dudz)
  + visibility channel (--use-vis)

Quickstart
----------
python train_fno_wind_terrain_eta_mask.py \
  --data-root data/wind_terrain_sobol_complex_perm_n_1000 \
  --iid --device cuda

Outputs
-------
- checkpoints/{run_tag}.pt           (best)
- checkpoints/{run_tag}__last.pt     (last)
- runs/{run_tag}/metrics.csv         (train/val logs)
- evals/{run_tag}__<split>.json      (summary)

"""

from __future__ import annotations
from __future__ import annotations

import json, time, random, re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


# ============================== USER CONFIG ==============================
# The defaults below are intentionally easy to edit.
# You can also override the most common options via CLI flags (see --help).

DATA_ROOT = Path("data") / "wind_terrain_sobol_complex_perm_n_1000"
RUN_TAG   = "fno_windterrain_u_eta_mask_coords_delta_iid"
DEVICE    = None  # "cuda" or "cpu" (None -> auto)

# Which split(s) to run
RUN_IID              = True
RUN_LOFO_WIND_ALL    = False   # leave-one-wind-family-out (u1/u2/u3/u4)
RUN_LOFO_TERRAIN_ALL = False   # leave-one-terrain-family-out (t1/t2/t3/t4)
RUN_LOFO_COMBO_ALL   = False   # leave-one-(u?,t?) combo out
RUN_RANGE_OOD        = False   # terrain-extremes split

# Core training configuration
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

    # ---- Masking ----
    "masked_loss": True,
    "masked_stats": True,
    "masked_metrics": True,

    # ---- Learn delta to baseline p0 if True ----
    "predict_delta": False,
    "baseline_file": "fno_aux/flat_ground.h5",  # relative to DATA_ROOT (your setup)

    # ---- Optional crop to safe window ----
    "crop_to_safe_x_window": False,
    "safe_x_window": (-450.0, 450.0),

    # ---- Debug ----
    "debug_overfit": False,
    "debug_n_cases": 8,
    "debug_constant_target": False,
    "debug_constant_re": 0.0,  # normalized space
    "debug_constant_im": 0.0,  # normalized space

    # ---- Wind input features ----
    "use_dudz": False,  # if True, adds dU/dz channel

    # ---- Optional visibility (vis) channel ----
    "use_visibility_channel": False,
    "visibility_soft": True,
    "visibility_tau": 0.02,
    "vis_method": "horizon_v1",
    "vis_eps": 1e-6,
    "vis_cache_tag": "features_vis_v1",
    "vis_cache_dir": "fno_aux/vis_cache",  # relative to DATA_ROOT; set None to cache next to cases
    "source_x": None,   # if None, read meta/src_x from each case
    "source_z": None,   # if None, read meta/src_z from each case

    # ---- Subset-per-epoch training (speed) ----
    "subset_per_epoch": True,
    "subset_frac": 0.30,

    # ---- Wind normalization ----
    "normalize_u": True,
    "u_norm_mode": "scale",
    "u_scale": 25.0,

    # ---- FNO architecture ----
    "in_channels": 5,   # inferred in train/predict/eval
    "out_channels": 2,
    "width": 96,
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

    # ---- Range-OOD split parameter (terrain-based) ----
    "extreme_pct": 15.0,
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
    return sorted((data_root / "cases").glob("case_*__u*__t*.h5"))

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

def load_h_profile(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h:
        xx = np.array(h["h_profile"]["x"][:], dtype=np.float32)
        hh = np.array(h["h_profile"]["h"][:], dtype=np.float32)
    return xx, hh

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

def parse_combo(h5_path: Path) -> Tuple[str, str, str]:
    m = re.search(r"__([uU]\d)__([tT]\d)\.h5$", h5_path.name)
    if not m:
        if "__" in h5_path.name:
            combo = h5_path.name.split("__", 1)[-1].split(".")[0]
            parts = combo.split("__")
            wf = parts[0].lower() if len(parts) > 0 else "u?"
            tf = parts[1].lower() if len(parts) > 1 else "t?"
            return combo, wf, tf
        return "u?__t?", "u?", "t?"
    wf = m.group(1).lower()
    tf = m.group(2).lower()
    return f"{wf}__{tf}", wf, tf


# ------------------------------ Geometry / Mask ------------------------------
def hx_on_xgrid(x_grid: np.ndarray, x_prof: np.ndarray, h_prof: np.ndarray) -> np.ndarray:
    if x_prof.shape == x_grid.shape and np.allclose(x_prof, x_grid):
        return h_prof.astype(np.float32)
    return np.interp(x_grid, x_prof, h_prof).astype(np.float32)

def air_mask_2d(xg: np.ndarray, zg: np.ndarray, hx_xgrid: np.ndarray) -> np.ndarray:
    return (zg[:, None] >= hx_xgrid[None, :])

def eta_norm_and_mask(xg: np.ndarray, zg: np.ndarray, hx_xgrid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Z = zg[:, None].astype(np.float32)
    HX = hx_xgrid[None, :].astype(np.float32)
    eta = Z - HX
    mask = (eta >= 0.0).astype(np.float32)
    z_range = float(zg.max() - zg.min() + 1e-8)
    eta_norm = (eta / z_range).astype(np.float32)
    return eta_norm, mask


# ------------------------------ Baseline p0 ------------------------------
def _baseline_path(data_root: Path, cfg: Dict[str, Any]) -> Path:
    rel = str(cfg.get("baseline_file", "fno_aux/flat_ground.h5"))
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
def _infer_in_channels(cfg: Dict[str, Any]) -> int:
    c = 5  # U + eta + mask + x + z
    if bool(cfg.get("use_dudz", False)):
        c += 1
    if bool(cfg.get("use_visibility_channel", False)):
        c += 1
    return c

def compute_stats_from_train(
    train_list: List[Path],
    cfg: Dict[str, Any],
    *,
    data_root: Path,
    p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    masked_stats = bool(cfg.get("masked_stats", True))
    predict_delta = bool(cfg.get("predict_delta", True))
    if predict_delta and p0 is None:
        p0 = load_baseline_p0(data_root, cfg)

    re_vals: List[np.ndarray] = []
    im_vals: List[np.ndarray] = []
    u_vals: List[np.ndarray] = []

    normalize_u = bool(cfg.get("normalize_u", True))
    u_norm_mode = str(cfg.get("u_norm_mode", "scale")).lower()

    for p in train_list:
        xg, zg = load_grids(p)

        x_prof, h_prof = load_h_profile(p)
        hx = hx_on_xgrid(xg, x_prof, h_prof)

        pre_raw, pim_raw = load_p_re_im(p)
        pre, pim = ensure_complex_zx(pre_raw, pim_raw, xg, zg)

        idx_x = None
        if cfg.get("crop_to_safe_x_window", False):
            xmin, xmax = cfg.get("safe_x_window", (-450.0, 450.0))
            idx_x = x_crop_indices(xg, float(xmin), float(xmax))
            if idx_x.size >= 16:
                xg = xg[idx_x]
                hx = hx[idx_x]
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

        if masked_stats:
            mask = air_mask_2d(xg, zg, hx)
            re_vals.append(dpre[mask].astype(np.float32))
            im_vals.append(dpim[mask].astype(np.float32))
        else:
            re_vals.append(dpre.reshape(-1).astype(np.float32))
            im_vals.append(dpim.reshape(-1).astype(np.float32))

        if normalize_u and u_norm_mode == "zscore":
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

        "masked_stats": masked_stats,
        "masked_loss": bool(cfg.get("masked_loss", True)),
        "masked_metrics": bool(cfg.get("masked_metrics", True)),

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


# ------------------------------ Visibility helpers ------------------------------
def _read_source_xz_from_case(h5_path: Path) -> Tuple[Optional[float], Optional[float]]:
    try:
        with h5py.File(h5_path, "r") as h:
            if "meta" in h:
                attrs = h["meta"].attrs
                sx = attrs.get("src_x", None)
                sz = attrs.get("src_z", None)

                def _to_float(v):
                    if v is None:
                        return None
                    if isinstance(v, (bytes, bytearray)):
                        v = v.decode("utf-8")
                    try:
                        return float(v)
                    except Exception:
                        return None

                return _to_float(sx), _to_float(sz)
    except Exception:
        pass
    return None, None

def _get_source_xz(h5_path: Path, cfg: Dict[str, Any]) -> Tuple[float, float]:
    xs = cfg.get("source_x", None)
    zs = cfg.get("source_z", None)
    if xs is not None and zs is not None:
        return float(xs), float(zs)
    sx2, sz2 = _read_source_xz_from_case(h5_path)
    if sx2 is not None and sz2 is not None:
        return float(sx2), float(sz2)
    return 0.0, 0.0

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def compute_visibility_horizon(
    xg: np.ndarray,
    zg: np.ndarray,
    hx_xgrid: np.ndarray,
    xs: float,
    zs: float,
    *,
    eps: float = 1e-6,
    soft: bool = True,
    tau: float = 0.02,
) -> np.ndarray:
    xg = np.asarray(xg, dtype=np.float32)
    zg = np.asarray(zg, dtype=np.float32)
    hx = np.asarray(hx_xgrid, dtype=np.float32)

    W = xg.shape[0]
    H = zg.shape[0]
    src_ix = int(np.argmin(np.abs(xg - np.float32(xs))))

    theta_hor = np.full((W,), -np.inf, dtype=np.float32)

    tmax = -np.inf
    for i in range(src_ix - 1, -1, -1):
        dx = abs(float(xg[i] - xs)) + float(eps)
        theta = np.arctan2(float(hx[i] - zs), dx).astype(np.float32)
        if theta > tmax:
            tmax = theta
        theta_hor[i] = np.float32(tmax)

    tmax = -np.inf
    for i in range(src_ix + 1, W):
        dx = abs(float(xg[i] - xs)) + float(eps)
        theta = np.arctan2(float(hx[i] - zs), dx).astype(np.float32)
        if theta > tmax:
            tmax = theta
        theta_hor[i] = np.float32(tmax)

    theta_hor[src_ix] = np.float32(-np.inf)

    dx = np.abs(xg[None, :] - np.float32(xs)) + np.float32(eps)
    theta_point = np.arctan2(zg[:, None] - np.float32(zs), dx).astype(np.float32)

    clearance = theta_point - theta_hor[None, :]

    if soft:
        t = max(1e-6, float(tau))
        vis = _sigmoid_stable(clearance / np.float32(t))
    else:
        vis = (clearance >= 0.0).astype(np.float32)

    air = (zg[:, None] >= hx[None, :]).astype(np.float32)
    vis *= air
    vis[:, src_ix] = air[:, src_ix]
    return vis.astype(np.float32)

def _vis_cache_path(case_path: Path, cfg: Dict[str, Any], data_root: Path) -> Path:
    tag = str(cfg.get("vis_cache_tag", "features_vis_v1"))
    cache_dir = cfg.get("vis_cache_dir", None)
    if cache_dir is None:
        return case_path.parent / f"{case_path.stem}.{tag}.npz"
    cache_dir = (data_root / str(cache_dir)).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{case_path.stem}.{tag}.npz"

def load_or_compute_vis(
    case_path: Path,
    xg: np.ndarray,
    zg: np.ndarray,
    hx_xgrid: np.ndarray,
    cfg: Dict[str, Any],
    *,
    data_root: Path,
) -> np.ndarray:
    xs, zs = _get_source_xz(case_path, cfg)
    eps = float(cfg.get("vis_eps", 1e-6))
    method = str(cfg.get("vis_method", "horizon_v1"))
    soft = bool(cfg.get("visibility_soft", True))
    tau  = float(cfg.get("visibility_tau", 0.02))

    cache = _vis_cache_path(case_path, cfg, data_root)
    case_mtime = float(case_path.stat().st_mtime)

    def meta_ok(meta: dict) -> bool:
        try:
            if meta.get("method") != method: return False
            if bool(meta.get("soft")) != soft: return False
            if abs(float(meta.get("xs")) - xs) > 1e-9: return False
            if abs(float(meta.get("zs")) - zs) > 1e-9: return False
            if abs(float(meta.get("eps")) - eps) > 1e-12: return False
            if abs(float(meta.get("tau")) - tau) > 1e-12: return False
            if int(meta.get("H")) != int(zg.shape[0]) or int(meta.get("W")) != int(xg.shape[0]): return False
            if abs(float(meta.get("x_min")) - float(xg.min())) > 1e-6: return False
            if abs(float(meta.get("x_max")) - float(xg.max())) > 1e-6: return False
            if abs(float(meta.get("z_min")) - float(zg.min())) > 1e-6: return False
            if abs(float(meta.get("z_max")) - float(zg.max())) > 1e-6: return False
            if abs(float(meta.get("case_mtime")) - case_mtime) > 1e-6: return False
            return True
        except Exception:
            return False

    if cache.exists():
        try:
            npz = np.load(cache, allow_pickle=True)
            vis = np.asarray(npz["vis"], dtype=np.float32)
            meta = dict(npz["meta"].item()) if "meta" in npz else {}
            if meta_ok(meta) and vis.shape == (zg.shape[0], xg.shape[0]):
                return vis
        except Exception:
            pass

    if method != "horizon_v1":
        raise ValueError(f"Unknown vis_method={method}")

    vis = compute_visibility_horizon(
        xg=xg, zg=zg, hx_xgrid=hx_xgrid, xs=xs, zs=zs,
        eps=eps, soft=soft, tau=tau,
    )

    meta = {
        "method": method,
        "soft": bool(soft),
        "tau": float(tau),
        "eps": float(eps),
        "xs": float(xs),
        "zs": float(zs),
        "H": int(zg.shape[0]),
        "W": int(xg.shape[0]),
        "x_min": float(xg.min()),
        "x_max": float(xg.max()),
        "z_min": float(zg.min()),
        "z_max": float(zg.max()),
        "case_mtime": float(case_mtime),
    }

    try:
        np.savez_compressed(cache, vis=vis.astype(np.float32), meta=np.array(meta, dtype=object))
    except Exception:
        pass

    return vis.astype(np.float32)


# ------------------------------ Dataset ------------------------------
class WindTerrainFNOResidualDataset(Dataset):
    """
    Returns:
      X: (Cin,H,W) float32  [U_grid_norm, (dudz_grid_norm), eta_norm, mask, (vis), x_norm, z_norm]
      Y: (2,H,W)   float32  normalized [?p_re, ?p_im]   (delta if predict_delta else total)
      mask: (H,W)  float32  air mask (1 air, 0 below ground)
    """
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

        self.predict_delta = bool(cfg.get("predict_delta", True))
        self.use_dudz = bool(cfg.get("use_dudz", False))
        self.use_vis  = bool(cfg.get("use_visibility_channel", False))

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
            return ((U - mu) / (sd + 1e-12)).astype(np.float32)
        return (U / (self.u_scale + 1e-12)).astype(np.float32)

    def _dudz_norm_1d(self, dudz: np.ndarray, z_range: float) -> np.ndarray:
        if not self.normalize_u:
            return dudz.astype(np.float32)
        if self.u_norm_mode == "zscore":
            sd = float(self.stats.get("u_std", 1.0))
            return (dudz * (z_range / (sd + 1e-12))).astype(np.float32)
        return (dudz * (z_range / (self.u_scale + 1e-12))).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
                hx = hx[idx_x]
                pre = pre[:, idx_x]
                pim = pim[:, idx_x]

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

        eta_norm, mask = eta_norm_and_mask(xg, zg, hx)

        z_u, U = load_u_profile(p)
        if len(U) != len(zg):
            U = np.interp(zg.astype(np.float64), z_u.astype(np.float64), U.astype(np.float64)).astype(np.float32)

        U_norm_1d = self._normalize_u_1d(U)
        U2d = np.broadcast_to(U_norm_1d[:, None], (H, W)).astype(np.float32)

        channels: List[np.ndarray] = [U2d]

        if self.use_dudz:
            dudz = np.gradient(U.astype(np.float64), zg.astype(np.float64)).astype(np.float32)
            z_range = float(zg.max() - zg.min() + 1e-8)
            dudz_norm_1d = self._dudz_norm_1d(dudz, z_range)
            dudz2d = np.broadcast_to(dudz_norm_1d[:, None], (H, W)).astype(np.float32)
            channels.append(dudz2d)

        x_norm_1d = (2.0 * (xg - xg.min()) / (xg.max() - xg.min() + 1e-8) - 1.0).astype(np.float32)
        z_norm_1d = (2.0 * (zg - zg.min()) / (zg.max() - zg.min() + 1e-8) - 1.0).astype(np.float32)
        x_norm = np.broadcast_to(x_norm_1d[None, :], (H, W)).astype(np.float32)
        z_norm = np.broadcast_to(z_norm_1d[:, None], (H, W)).astype(np.float32)

        if self.use_vis:
            vis = load_or_compute_vis(
                case_path=p, xg=xg, zg=zg, hx_xgrid=hx, cfg=self.cfg, data_root=self.data_root
            )
            channels += [eta_norm.astype(np.float32), mask.astype(np.float32), vis.astype(np.float32), x_norm, z_norm]
        else:
            channels += [eta_norm.astype(np.float32), mask.astype(np.float32), x_norm, z_norm]

        X = np.stack(channels, axis=0).astype(np.float32)
        return {"X": X, "Y": Y, "mask": mask.astype(np.float32), "path": str(p)}

def collate_fields(batch: List[Dict[str, Any]]):
    X = torch.from_numpy(np.stack([b["X"] for b in batch], axis=0))
    Y = torch.from_numpy(np.stack([b["Y"] for b in batch], axis=0))
    M = torch.from_numpy(np.stack([b["mask"] for b in batch], axis=0))
    paths = [b["path"] for b in batch]
    return {"X": X, "Y": Y, "mask": M, "paths": paths}


# ------------------------------ FNO model ------------------------------
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
    mae  = float(np.mean(np.abs(diff)))
    return rmse, mae


# ------------------------------ Splits ------------------------------
def make_splits_iid(data_root: Path, cfg=CFG):
    cases = find_cases(data_root)
    assert cases, f"No cases found under {data_root}/cases"

    by_combo: Dict[str, List[Path]] = {}
    for p in cases:
        combo, _, _ = parse_combo(p)
        by_combo.setdefault(combo, []).append(p)

    rng = np.random.default_rng(int(cfg["seed"]))
    iid_dir = data_root / "splits" / "iid"
    if (iid_dir / "train.txt").exists():
        print("[splits] IID already exists -> skip")
        return

    train, val, test = [], [], []
    for combo, lst in by_combo.items():
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

def make_splits_lofo_wind(data_root: Path, cfg=CFG):
    cases = find_cases(data_root)
    assert cases, f"No cases found under {data_root}/cases"
    by_w: Dict[str, List[Path]] = {}
    for p in cases:
        _, wf, _ = parse_combo(p)
        by_w.setdefault(wf, []).append(p)

    rng = np.random.default_rng(int(cfg["seed"]))
    lofo_dir = data_root / "splits" / "lofo_wind"
    for wf in sorted(by_w):
        fam_dir = lofo_dir / wf
        if (fam_dir / "train.txt").exists():
            continue
        test = list(by_w[wf])
        train_pool = [p for w2, lst in by_w.items() if w2 != wf for p in lst]
        rng.shuffle(train_pool)
        nva = max(1, int(0.15 * len(train_pool)))
        val = train_pool[:nva]
        train = train_pool[nva:]
        write_list(fam_dir / "train.txt", train)
        write_list(fam_dir / "val.txt",   val)
        write_list(fam_dir / "test.txt",  test)
        print(f"[splits] Wrote LOFO_WIND/{wf}: {len(train)}/{len(val)}/{len(test)}")

def make_splits_lofo_terrain(data_root: Path, cfg=CFG):
    cases = find_cases(data_root)
    assert cases, f"No cases found under {data_root}/cases"
    by_t: Dict[str, List[Path]] = {}
    for p in cases:
        _, _, tf = parse_combo(p)
        by_t.setdefault(tf, []).append(p)

    rng = np.random.default_rng(int(cfg["seed"]))
    lofo_dir = data_root / "splits" / "lofo_terrain"
    for tf in sorted(by_t):
        fam_dir = lofo_dir / tf
        if (fam_dir / "train.txt").exists():
            continue
        test = list(by_t[tf])
        train_pool = [p for t2, lst in by_t.items() if t2 != tf for p in lst]
        rng.shuffle(train_pool)
        nva = max(1, int(0.15 * len(train_pool)))
        val = train_pool[:nva]
        train = train_pool[nva:]
        write_list(fam_dir / "train.txt", train)
        write_list(fam_dir / "val.txt",   val)
        write_list(fam_dir / "test.txt",  test)
        print(f"[splits] Wrote LOFO_TERRAIN/{tf}: {len(train)}/{len(val)}/{len(test)}")

def make_splits_lofo_combo(data_root: Path, cfg=CFG):
    cases = find_cases(data_root)
    assert cases, f"No cases found under {data_root}/cases"
    by_c: Dict[str, List[Path]] = {}
    for p in cases:
        combo, _, _ = parse_combo(p)
        by_c.setdefault(combo, []).append(p)

    rng = np.random.default_rng(int(cfg["seed"]))
    lofo_dir = data_root / "splits" / "lofo_combo"
    for combo in sorted(by_c):
        fam_dir = lofo_dir / combo
        if (fam_dir / "train.txt").exists():
            continue
        test = list(by_c[combo])
        train_pool = [p for c2, lst in by_c.items() if c2 != combo for p in lst]
        rng.shuffle(train_pool)
        nva = max(1, int(0.15 * len(train_pool)))
        val = train_pool[:nva]
        train = train_pool[nva:]
        write_list(fam_dir / "train.txt", train)
        write_list(fam_dir / "val.txt",   val)
        write_list(fam_dir / "test.txt",  test)
        print(f"[splits] Wrote LOFO_COMBO/{combo}: {len(train)}/{len(val)}/{len(test)}")

def make_splits_range_ood(data_root: Path, cfg=CFG):
    cases = find_cases(data_root)
    assert cases, f"No cases found under {data_root}/cases"
    rdir = data_root / "splits" / "range_ood"
    if (rdir / "train.txt").exists():
        print("[splits] Range-OOD already exists -> skip")
        return

    heights, slopes = [], []
    for p in cases:
        xg, _ = load_grids(p)
        x_prof, h_prof = load_h_profile(p)
        hx = hx_on_xgrid(xg, x_prof, h_prof)
        heights.append(float(np.max(hx)))
        dh = np.gradient(hx, xg)
        slopes.append(float(np.max(np.abs(dh))))

    heights = np.array(heights, dtype=np.float32)
    slopes  = np.array(slopes, dtype=np.float32)
    h0 = (heights - heights.min()) / (np.ptp(heights) + 1e-8)
    s0 = (slopes  - slopes.min())  / (np.ptp(slopes)  + 1e-8)
    sc = np.maximum(h0, s0)

    order = np.argsort(sc)[::-1]
    cases_arr = np.array(cases)
    extreme_pct = float(cfg.get("extreme_pct", 15.0))
    extreme_k = max(1, int(len(cases) * (extreme_pct / 100.0)))

    test = list(cases_arr[order[:extreme_k]])
    rest = list(cases_arr[order[extreme_k:]])

    rng = np.random.default_rng(int(cfg["seed"]))
    rng.shuffle(rest)
    nva = max(1, int(0.15 * len(rest)))
    val = rest[:nva]
    train = rest[nva:]

    write_list(rdir / "train.txt", train)
    write_list(rdir / "val.txt",   val)
    write_list(rdir / "test.txt",  test)
    print(f"[splits] Wrote Range-OOD: {len(train)}/{len(val)}/{len(test)} (extreme_pct={extreme_pct})")


# ------------------------------ Train / Eval ------------------------------
def train_run(
    data_root: Path,
    split: str,
    run_tag: str,
    lofo_key: str | None = None,
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
    elif split == "range_ood":
        sp_dir = splits_dir / "range_ood"
    elif split == "lofo_wind":
        assert lofo_key is not None
        sp_dir = splits_dir / "lofo_wind" / lofo_key
    elif split == "lofo_terrain":
        assert lofo_key is not None
        sp_dir = splits_dir / "lofo_terrain" / lofo_key
    elif split == "lofo_combo":
        assert lofo_key is not None
        sp_dir = splits_dir / "lofo_combo" / lofo_key
    else:
        raise ValueError("split must be one of: iid, range_ood, lofo_wind, lofo_terrain, lofo_combo")

    train_list = read_list(sp_dir / "train.txt")
    val_list   = read_list(sp_dir / "val.txt")

    if cfg.get("debug_overfit", False):
        n_dbg = int(cfg.get("debug_n_cases", 8))
        n_dbg = max(1, min(n_dbg, len(train_list)))
        train_list = train_list[:n_dbg]
        val_list = list(train_list)
        print(f"[debug] OVERFIT mode ON -> using {len(train_list)} train/val cases")

    # ---- CHANGE #1: only load baseline if predict_delta ----
    predict_delta = bool(cfg.get("predict_delta", True))
    p0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    if predict_delta:
        p0 = load_baseline_p0(data_root, cfg)
        xg0, zg0, _, _ = p0
        xg1, zg1 = load_grids(train_list[0])
        _assert_same_grid(xg1, zg1, xg0, zg0, str(train_list[0]), "baseline_p0")
        print(f"[train] Using DELTA target (baseline_file={_baseline_path(data_root, cfg)})")
    else:
        print("[train] Using TOTAL target (no baseline subtraction)")

    aux_dir = data_root / "fno_aux"
    ensure_dir(aux_dir)

    dudz_suffix = "dudz" if bool(cfg.get("use_dudz", False)) else "u"
    # ---- CHANGE #3: stats filename reflects delta vs full ----
    tgt_suffix = "delta" if predict_delta else "full"
    stats_path = aux_dir / f"stats_complex_windterrain_{tgt_suffix}_{dudz_suffix}.json"

    def _stats_compatible(st: Dict[str, Any], cfg_local: Dict[str, Any]) -> bool:
        # ---- CHANGE #2: baseline_file compared only when delta is enabled ----
        if bool(st.get("predict_delta", True)) != bool(cfg_local.get("predict_delta", True)):
            return False
        if bool(cfg_local.get("predict_delta", True)):
            if str(st.get("baseline_file", "")) != str(cfg_local.get("baseline_file", "")):
                return False

        if bool(st.get("use_dudz", False)) != bool(cfg_local.get("use_dudz", False)):
            return False
        if bool(st.get("masked_stats", True)) != bool(cfg_local.get("masked_stats", True)):
            return False
        if bool(st.get("normalize_u", True)) != bool(cfg_local.get("normalize_u", True)):
            return False
        if str(st.get("u_norm_mode", "scale")) != str(cfg_local.get("u_norm_mode", "scale")):
            return False
        if float(st.get("u_scale", 25.0)) != float(cfg_local.get("u_scale", 25.0)):
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

    tr_ds = WindTerrainFNOResidualDataset(train_list, stats, cfg, data_root=data_root, p0=p0)
    va_ds = WindTerrainFNOResidualDataset(val_list,   stats, cfg, data_root=data_root, p0=p0)

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

    # ---- CHANGE #4: keep this inside train_run (closure) ----
    def make_train_loader_for_epoch(epoch: int) -> DataLoader:
        if not bool(cfg.get("subset_per_epoch", False)):
            return tr_ld

        frac = float(cfg.get("subset_frac", 0.30))
        frac = min(max(frac, 0.0), 1.0)

        n_total = len(tr_ds)
        n_sub = max(1, int(round(frac * n_total)))
        n_sub = min(n_sub, n_total)

        rng = np.random.default_rng(int(cfg.get("seed", 0)) + int(epoch))
        idx = rng.choice(n_total, size=n_sub, replace=False)

        sub_ds = Subset(tr_ds, idx.tolist())

        return DataLoader(
            sub_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            drop_last=True,
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
            "lofo_key": lofo_key,
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

    print(f"[train] split={split} lofo_key={lofo_key} device={device_info_string(device)}")
    print(f"[train] Ntrain={len(train_list)} Nval={len(val_list)} batch={cfg['batch_size']} amp={use_amp}")
    print(f"[train] masked_loss={cfg.get('masked_loss', True)} masked_stats={cfg.get('masked_stats', True)} masked_metrics={cfg.get('masked_metrics', True)}")
    print(f"[train] predict_delta={cfg.get('predict_delta', True)} baseline_file={cfg.get('baseline_file', '')}")
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

        tr_ld_epoch = make_train_loader_for_epoch(epoch)
        if bool(cfg.get("subset_per_epoch", False)):
            try:
                print(f"  using subset: {len(tr_ld_epoch.dataset)}/{len(tr_ds)} train cases (subset_frac={float(cfg.get('subset_frac',0.0)):.2f})")
            except Exception:
                pass

        model.train()
        tr_loss = 0.0
        nb = 0
        for batch in tr_ld_epoch:
            X = batch["X"].to(device, non_blocking=True)
            Y = batch["Y"].to(device, non_blocking=True)
            M = batch["mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                assert scaler is not None
                with torch.amp.autocast(device_type=device_type, enabled=True):
                    pred = model(X)
                    loss = masked_mse_field(pred, Y, M) if cfg.get("masked_loss", True) else torch.mean((pred - Y) ** 2)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(X)
                loss = masked_mse_field(pred, Y, M) if cfg.get("masked_loss", True) else torch.mean((pred - Y) ** 2)
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
                        loss = masked_mse_field(pred, Y, M) if cfg.get("masked_loss", True) else torch.mean((pred - Y) ** 2)
                else:
                    pred = model(X)
                    loss = masked_mse_field(pred, Y, M) if cfg.get("masked_loss", True) else torch.mean((pred - Y) ** 2)

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
    x_prof, h_prof = load_h_profile(path)
    hx = hx_on_xgrid(xg, x_prof, h_prof)

    if bool(cfg.get("predict_delta", True)) and p0 is None:
        p0 = load_baseline_p0(data_root, cfg)

    idx_x = None
    if cfg.get("crop_to_safe_x_window", False):
        xmin, xmax = cfg.get("safe_x_window", (-450.0, 450.0))
        idx_x = x_crop_indices(xg, float(xmin), float(xmax))
        if idx_x.size >= 16:
            xg = xg[idx_x]
            hx = hx[idx_x]
            if p0 is not None:
                xg0, zg0, p0_re, p0_im = p0
                p0 = (xg0[idx_x], zg0, p0_re[:, idx_x], p0_im[:, idx_x])

    if p0 is not None:
        xg0, zg0, _, _ = p0
        _assert_same_grid(xg, zg, xg0, zg0, str(path), "baseline_p0")

    H, W = len(zg), len(xg)

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

    U2d = np.broadcast_to(U_norm_1d[:, None], (H, W)).astype(np.float32)

    eta_norm, mask = eta_norm_and_mask(xg, zg, hx)

    x_norm_1d = (2.0 * (xg - xg.min()) / (xg.max() - xg.min() + 1e-8) - 1.0).astype(np.float32)
    z_norm_1d = (2.0 * (zg - zg.min()) / (zg.max() - zg.min() + 1e-8) - 1.0).astype(np.float32)
    x_norm = np.broadcast_to(x_norm_1d[None, :], (H, W)).astype(np.float32)
    z_norm = np.broadcast_to(z_norm_1d[:, None], (H, W)).astype(np.float32)

    channels: List[np.ndarray] = [U2d]
    if bool(cfg.get("use_dudz", False)):
        dudz = np.gradient(U.astype(np.float64), zg.astype(np.float64)).astype(np.float32)
        z_range = float(zg.max() - zg.min() + 1e-8)
        if not normalize_u:
            dudz_norm_1d = dudz.astype(np.float32)
        elif u_norm_mode == "zscore":
            sd = float(stats.get("u_std", 1.0))
            dudz_norm_1d = (dudz * (z_range / (sd + 1e-12))).astype(np.float32)
        else:
            dudz_norm_1d = (dudz * (z_range / (u_scale + 1e-12))).astype(np.float32)
        channels.append(np.broadcast_to(dudz_norm_1d[:, None], (H, W)).astype(np.float32))

    channels += [eta_norm.astype(np.float32), mask.astype(np.float32), x_norm, z_norm]
    X = np.stack(channels, axis=0)[None, ...].astype(np.float32)
    X_t = torch.from_numpy(X).to(device)

    pred_std = model(X_t).float().cpu().numpy()[0]
    dpre = pred_std[0] * float(stats["p_re_std"]) + float(stats["p_re_mean"])
    dpim = pred_std[1] * float(stats["p_im_std"]) + float(stats["p_im_mean"])

    if bool(cfg.get("predict_delta", True)):
        assert p0 is not None
        _, _, p0_re, p0_im = p0
        ppre = dpre + p0_re
        ppim = dpim + p0_im
    else:
        ppre, ppim = dpre, dpim

    return ppre.astype(np.float32), ppim.astype(np.float32)

def eval_run(
    data_root: Path,
    split: str,
    run_tag: str,
    ckpt: Path,
    lofo_key: str | None = None,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = dict(state.get("cfg", CFG))
    cfg["in_channels"] = _infer_in_channels(cfg)
    stats = state["stats"]

    man = load_manifest(data_root)
    if cfg.get("crop_to_safe_x_window", False) and "safe_x_window" in man:
        cfg["safe_x_window"] = tuple(man["safe_x_window"])

    splits_dir = data_root / "splits"
    if split == "iid":
        sp_dir = splits_dir / "iid"
    elif split == "range_ood":
        sp_dir = splits_dir / "range_ood"
    elif split == "lofo_wind":
        assert lofo_key is not None
        sp_dir = splits_dir / "lofo_wind" / lofo_key
    elif split == "lofo_terrain":
        assert lofo_key is not None
        sp_dir = splits_dir / "lofo_terrain" / lofo_key
    elif split == "lofo_combo":
        assert lofo_key is not None
        sp_dir = splits_dir / "lofo_combo" / lofo_key
    else:
        raise ValueError("split must be one of: iid, range_ood, lofo_wind, lofo_terrain, lofo_combo")

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

    p0 = load_baseline_p0(data_root, cfg) if bool(cfg.get("predict_delta", True)) else None
    if p0 is not None:
        print(f"[eval] Using baseline reconstruction (baseline_file={_baseline_path(data_root, cfg)})")

    use_masked = bool(cfg.get("masked_metrics", True))

    all_rmse_re, all_rmse_im = [], []
    all_mae_re, all_mae_im = [], []
    all_ssim_re, all_ssim_im = [], []

    per_family: Dict[str, Dict[str, List[float]]] = {}
    per_wind: Dict[str, Dict[str, List[float]]] = {}
    per_terrain: Dict[str, Dict[str, List[float]]] = {}

    for path in test_list:
        xg, zg = load_grids(path)
        x_prof, h_prof = load_h_profile(path)
        hx = hx_on_xgrid(xg, x_prof, h_prof)

        gt_re_raw, gt_im_raw = load_p_re_im(path)
        gt_re, gt_im = ensure_complex_zx(gt_re_raw, gt_im_raw, xg, zg)

        if cfg.get("crop_to_safe_x_window", False):
            xmin, xmax = cfg.get("safe_x_window", (-450.0, 450.0))
            idx_x = x_crop_indices(xg, float(xmin), float(xmax))
            if idx_x.size >= 16:
                xg = xg[idx_x]
                hx = hx[idx_x]
                gt_re = gt_re[:, idx_x]
                gt_im = gt_im[:, idx_x]

        pr_re, pr_im = predict_case(model, device, path, stats, cfg, data_root=data_root, p0=p0)

        if use_masked:
            mask2d = air_mask_2d(xg, zg, hx)
            rmse_re, mae_re = _masked_error_stats(pr_re, gt_re, mask2d)
            rmse_im, mae_im = _masked_error_stats(pr_im, gt_im, mask2d)

            a_re = np.where(mask2d, pr_re, 0.0)
            b_re = np.where(mask2d, gt_re, 0.0)
            a_im = np.where(mask2d, pr_im, 0.0)
            b_im = np.where(mask2d, gt_im, 0.0)
            ssim_re = ssim_global(a_re, b_re)
            ssim_im = ssim_global(a_im, b_im)
        else:
            rmse_re = float(np.sqrt(np.mean((pr_re - gt_re) ** 2)))
            rmse_im = float(np.sqrt(np.mean((pr_im - gt_im) ** 2)))
            mae_re  = float(np.mean(np.abs(pr_re - gt_re)))
            mae_im  = float(np.mean(np.abs(pr_im - gt_im)))
            ssim_re = ssim_global(pr_re, gt_re)
            ssim_im = ssim_global(pr_im, gt_im)

        all_rmse_re.append(rmse_re); all_rmse_im.append(rmse_im)
        all_mae_re.append(mae_re);   all_mae_im.append(mae_im)
        all_ssim_re.append(ssim_re); all_ssim_im.append(ssim_im)

        combo, wf, tf = parse_combo(path)

        def _acc(dct: Dict[str, Dict[str, List[float]]], key: str):
            dd = dct.setdefault(key, {"rmse_re": [], "rmse_im": [], "mae_re": [], "mae_im": [], "ssim_re": [], "ssim_im": []})
            dd["rmse_re"].append(rmse_re); dd["rmse_im"].append(rmse_im)
            dd["mae_re"].append(mae_re);   dd["mae_im"].append(mae_im)
            dd["ssim_re"].append(ssim_re); dd["ssim_im"].append(ssim_im)

        _acc(per_family, combo)
        _acc(per_wind, wf)
        _acc(per_terrain, tf)

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
            "mae_re":  summarize(all_mae_re),
            "mae_im":  summarize(all_mae_im),
            "ssim_re": summarize(all_ssim_re),
            "ssim_im": summarize(all_ssim_im),
        },
        "per_family": {k: {m: summarize(v) for m, v in d.items()} for k, d in per_family.items()},
        "per_wind":   {k: {m: summarize(v) for m, v in d.items()} for k, d in per_wind.items()},
        "per_terrain":{k: {m: summarize(v) for m, v in d.items()} for k, d in per_terrain.items()},

        "n_cases": len(test_list),
        "split": split,
        "lofo_key": lofo_key,
        "run_tag": run_tag,
        "ckpt": str(ckpt),

        "masked_metrics": use_masked,
        "masked_loss": bool(cfg.get("masked_loss", True)),
        "masked_stats": bool(cfg.get("masked_stats", True)),

        "predict_delta": bool(cfg.get("predict_delta", True)),
        "baseline_file": str(cfg.get("baseline_file", "")),

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
    out_path = out_dir / f"{run_tag}__{split}{('_'+lofo_key) if (split.startswith('lofo') and lofo_key) else ''}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[eval] wrote summary to {out_path}")


# ------------------------------ Autorun ------------------------------
def run_everything():
    data_root = Path(DATA_ROOT)
    print(f"[autorun] Using DATA_ROOT: {data_root}")

    if not (data_root / "splits" / "iid" / "train.txt").exists():
        print("[autorun] IID splits not found -> creating...")
        make_splits_iid(data_root, CFG)
    else:
        print("[autorun] IID splits already present -> skipping.")

    if RUN_LOFO_WIND_ALL:
        make_splits_lofo_wind(data_root, CFG)
    if RUN_LOFO_TERRAIN_ALL:
        make_splits_lofo_terrain(data_root, CFG)
    if RUN_LOFO_COMBO_ALL:
        make_splits_lofo_combo(data_root, CFG)
    if RUN_RANGE_OOD:
        make_splits_range_ood(data_root, CFG)

    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device_info_string(device)}")

    if RUN_IID:
        tag = RUN_TAG
        print(f"[autorun] Training IID as '{tag}'")
        train_run(data_root, split="iid", run_tag=tag, device=device)
        print(f"[autorun] Evaluating IID (BEST) as '{tag}'")
        eval_run(data_root, split="iid", run_tag=tag, ckpt=Path(f"checkpoints/{tag}.pt"), device=device)

    if RUN_RANGE_OOD:
        tag = "fno_windterrain_u_eta_mask_coords_delta_range_ood"
        print(f"[autorun] Training Range-OOD as '{tag}'")
        train_run(data_root, split="range_ood", run_tag=tag, device=device)
        print(f"[autorun] Evaluating Range-OOD (BEST) as '{tag}'")
        eval_run(data_root, split="range_ood", run_tag=tag, ckpt=Path(f"checkpoints/{tag}.pt"), device=device)

    if RUN_LOFO_WIND_ALL:
        wind_fams = sorted({parse_combo(p)[1] for p in find_cases(data_root)})
        for wf in wind_fams:
            tag = f"fno_windterrain_u_eta_mask_coords_delta_lofo_wind_{wf}"
            print(f"[autorun] Training LOFO_WIND ({wf}) as '{tag}'")
            train_run(data_root, split="lofo_wind", lofo_key=wf, run_tag=tag, device=device)
            print(f"[autorun] Evaluating LOFO_WIND ({wf}) as '{tag}'")
            eval_run(data_root, split="lofo_wind", lofo_key=wf, run_tag=tag, ckpt=Path(f"checkpoints/{tag}.pt"), device=device)

    if RUN_LOFO_TERRAIN_ALL:
        terr_fams = sorted({parse_combo(p)[2] for p in find_cases(data_root)})
        for tf in terr_fams:
            tag = f"fno_windterrain_u_eta_mask_coords_delta_lofo_terrain_{tf}"
            print(f"[autorun] Training LOFO_TERRAIN ({tf}) as '{tag}'")
            train_run(data_root, split="lofo_terrain", lofo_key=tf, run_tag=tag, device=device)
            print(f"[autorun] Evaluating LOFO_TERRAIN ({tf}) as '{tag}'")
            eval_run(data_root, split="lofo_terrain", lofo_key=tf, run_tag=tag, ckpt=Path(f"checkpoints/{tag}.pt"), device=device)

    if RUN_LOFO_COMBO_ALL:
        combos = sorted({parse_combo(p)[0] for p in find_cases(data_root)})
        for combo in combos:
            tag = f"fno_windterrain_u_eta_mask_coords_delta_lofo_combo_{combo}"
            print(f"[autorun] Training LOFO_COMBO ({combo}) as '{tag}'")
            train_run(data_root, split="lofo_combo", lofo_key=combo, run_tag=tag, device=device)
            print(f"[autorun] Evaluating LOFO_COMBO ({combo}) as '{tag}'")
            eval_run(data_root, split="lofo_combo", lofo_key=combo, run_tag=tag, ckpt=Path(f"checkpoints/{tag}.pt"), device=device)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate FNO for joint wind+terrain surrogate.")
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT), help="Dataset root folder (contains cases/, splits/, fno_aux/...).")
    p.add_argument("--run-tag", type=str, default=RUN_TAG, help="Name used for checkpoints and run folder.")
    p.add_argument("--device", type=str, default=DEVICE, help="cuda / cpu. If omitted, auto-select.")

    # Split selection
    p.add_argument("--iid", action="store_true", help="Run IID train/eval.")
    p.add_argument("--lofo-wind", action="store_true", help="Run LOFO wind families.")
    p.add_argument("--lofo-terrain", action="store_true", help="Run LOFO terrain families.")
    p.add_argument("--lofo-combo", action="store_true", help="Run LOFO (u,t) combos.")
    p.add_argument("--range-ood", action="store_true", help="Run terrain-range OOD split.")

    # Common toggles
    p.add_argument("--direct", action="store_true", help="Train direct p instead of residual Δp to baseline p0.")
    p.add_argument("--baseline-file", type=str, default=str(CFG.get("baseline_file", "fno_aux/flat_ground.h5")),
                   help="Relative to data-root, used when training residuals.")
    p.add_argument("--use-dudz", action="store_true", help="Add dU/dz as an extra input channel.")
    p.add_argument("--use-vis", action="store_true", help="Add visibility channel (requires meta/src_x, meta/src_z in cases).")

    # Convenience
    p.add_argument("--epochs", type=int, default=int(CFG.get("epochs", 4000)), help="Number of epochs.")
    p.add_argument("--lr", type=float, default=float(CFG.get("lr", 1e-4)), help="Learning rate.")
    p.add_argument("--subset-frac", type=float, default=float(CFG.get("subset_frac", 0.30)),
                   help="If subset_per_epoch=True, fraction of training cases per epoch.")
    p.add_argument("--no-subset", action="store_true", help="Disable subset_per_epoch.")
    p.add_argument("--amp", action="store_true", help="Enable AMP.")
    p.add_argument("--no-amp", action="store_true", help="Disable AMP.")

    return p.parse_args()


def main() -> None:
    global DATA_ROOT, RUN_TAG, DEVICE
    global RUN_IID, RUN_LOFO_WIND_ALL, RUN_LOFO_TERRAIN_ALL, RUN_LOFO_COMBO_ALL, RUN_RANGE_OOD
    args = _parse_args()

    DATA_ROOT = Path(args.data_root)
    RUN_TAG = args.run_tag
    DEVICE = args.device

    # If user passed any split flags, use them; otherwise keep defaults from USER CONFIG.
    any_split_flag = any([args.iid, args.lofo_wind, args.lofo_terrain, args.lofo_combo, args.range_ood])
    if any_split_flag:
        RUN_IID = bool(args.iid)
        RUN_LOFO_WIND_ALL = bool(args.lofo_wind)
        RUN_LOFO_TERRAIN_ALL = bool(args.lofo_terrain)
        RUN_LOFO_COMBO_ALL = bool(args.lofo_combo)
        RUN_RANGE_OOD = bool(args.range_ood)

    # Apply common overrides into CFG
    if args.direct:
        CFG["predict_delta"] = False
    else:
        # keep default (often True); explicitness helps in CLI use
        CFG["predict_delta"] = bool(CFG.get("predict_delta", True))

    CFG["baseline_file"] = args.baseline_file
    CFG["use_dudz"] = bool(args.use_dudz)
    CFG["use_visibility_channel"] = bool(args.use_vis)

    CFG["epochs"] = int(args.epochs)
    CFG["lr"] = float(args.lr)
    CFG["subset_frac"] = float(args.subset_frac)
    if args.no_subset:
        CFG["subset_per_epoch"] = False

    if args.no_amp:
        CFG["amp"] = False
    elif args.amp:
        CFG["amp"] = True

    run_everything()


if __name__ == "__main__":
    main()
