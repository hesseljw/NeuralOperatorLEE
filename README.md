# Neural-Operator Surrogates for Outdoor Acoustics (LEE-FDTD)

This repository contains training scripts for **DeepONet** and **Fourier Neural Operator (FNO)** surrogates that learn mappings from:
- parameterized **wind profiles** `U(z)` and/or
- parameterized **terrain profiles** `h(x)`

to the **complex harmonic acoustic pressure field** `p(x,z)` on a fixed 2D Region of Interest (ROI) grid.

The scripts are aligned with the accompanying paper draft:
**“Towards Neural-operator Surrogates for Outdoor Acoustics using Parameterized Wind Profiles and Terrain.”**

---

## What’s in here

### Training scripts (GitHub-ready)

| Script | What it trains | Input → Output |
|---|---|---|
| `train_deeponet_wind_gridtrunk.py` | DeepONet (wind-only) | `U(z)` → `p(x,z)` (complex) |
| `train_deeponet_terrain_gridtrunk.py` | DeepONet (terrain-only) | `h(x)` → `p(x,z)` (complex) |
| `train_fno_wind_coords.py` | FNO (wind-only) | grid features from `U(z)` + coords → `p(x,z)` (complex) |
| `train_fno_terrain_eta_mask_phys.py` | FNO (terrain-only) | `[η(z-h), air_mask, x, z]` → **Δp** (complex) |
| `train_fno_wind_terrain_eta_mask.py` | FNO (wind + terrain) | `[U(z), η(z-h), air_mask, x, z]` → `p` or **Δp** (complex) |

> Each script includes a **USER CONFIG** block at the top (easy to edit) and minimal CLI flags (run `--help`).

---

## Data layout (expected)

Each dataset root folder should look like:

```
data/<DATASET_ROOT>/
  cases/
    case_000001__u1__t3.h5
    case_000002__u2__t1.h5
    ...
  splits/
    iid/
      train.txt
      val.txt
      test.txt
    lofo_wind/ ... (optional)
    lofo_terrain/ ... (optional)
    lofo_combo/ ... (optional)
    range_ood/ ... (optional)
  fno_aux/
    flat_ground.h5          # baseline p0 for residual learning
    ...                     # (optional) caches, stats, etc.
  manifest.json             # (optional)
```

Each `case_*.h5` typically contains:
- `x_grid`, `z_grid`
- `p_re`, `p_im`
- `u_profile/{z,U}` for wind datasets
- `h_profile/{x,h}` for terrain datasets

---

## Residual learning (Δp) and masking

Some scripts train the residual to a baseline solution:

- **Terrain residual:** `Δp = p_terrain − p_flat`
- **Joint residual:** `Δp = p(U,h) − p0` (p0 is a flat-ground baseline in the joint dataset)

Training/evaluation can be **masked** to the air region:
`Ω_air = {(x,z): z ≥ h(x)}`

This avoids penalizing the model inside the ground and stabilizes metrics.

---

## Quickstart

### FNO wind-only
```bash
python train_fno_wind_coords.py --data-root data/wind_sobol_complex_n_1000 --iid --device cuda
```

### FNO terrain-only (η + mask, residual to flat-ground)
```bash
python train_fno_terrain_eta_mask_phys.py --data-root data/terrain_sobol_complex_n_1000 --iid --device cuda
```

### FNO wind + terrain (joint)
```bash
python train_fno_wind_terrain_eta_mask.py --data-root data/wind_terrain_sobol_complex_perm_n_1000 --iid --device cuda
```

---

## Outputs

Scripts write (by default):
- `checkpoints/<run_tag>.pt` (best)
- `checkpoints/<run_tag>__last.pt` (last)
- `runs/<run_tag>/metrics.csv`
- `evals/<run_tag>__<split>.json`

---

## Notes
- The scripts assume **PyTorch** and HDF5 (`h5py`).
- For mixed precision, FFT parts are forced to float32 where needed to avoid cuFFT fp16 limitations (common on some GPUs).

---

## Citation

If you use this code, please cite the associated paper (preprint / submission information to be added here).
