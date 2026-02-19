# NeuralOperatorLEE

Training scripts for **DeepONet** and **Fourier Neural Operator (FNO)** surrogates for outdoor acoustics generated with a **Linearized Euler Equations (LEE) FDTD** solver.

Models learn mappings from
- parameterized **wind profiles** `U(z)` and/or
- parameterized **terrain profiles** `h(x)`
to the **complex harmonic acoustic pressure field** on a fixed 2D ROI grid.

The scripts are aligned with the paper draft:
**“Towards Neural-operator Surrogates for Outdoor Acoustics using Parameterized Wind Profiles and Terrain.”**

---

## What’s in this repo

### Training scripts

All scripts can train either:
- **direct pressure** `p(x,z)` *(2 channels: Re/Im)*, or
- **residual pressure** `Δp(x,z)` to a provided baseline *(Re/Im)*.

Residual targets are reconstructed as `p_pred = p0 + Δp_pred` (wind / joint) or `p_pred = p_flat + Δp_pred` (terrain).

> **Where to switch p vs Δp?**  
> Each script has a **USER CONFIG** block at the top. Flip the indicated boolean to choose the target.  
> The joint FNO script also exposes a `--direct` CLI flag.

| Script | Setting for **p vs Δp** | Default |
|---|---|---|
| `train_deeponet_wind_gridtrunk.py` | `predict_delta` (True → Δp to baseline `p0`) | `False` (direct `p`) |
| `train_deeponet_terrain_eta_fourier.py` | `use_flat_residual` (True → Δp to `p_flat`) | `True` (Δp) |
| `train_fno_wind_coords.py` | `predict_delta` (True → Δp to baseline `p0`) | `False` (direct `p`) |
| `train_fno_terrain_eta_mask_phys.py` | `use_flat_residual` (True → Δp to `p_flat`) | `False` (direct `p`) |
| `train_fno_joint_wind_terrain_eta_mask.py` | `predict_delta` / `--direct` | `True` (Δp) |

---

## Masking (terrain + joint)

Terrain cases use an **air mask** to compute loss/metrics only in the air region:
`Ω_air = {(x,z): z ≥ h(x)}`

This avoids penalizing predictions inside the ground and stabilizes training.

---

## Expected data layout

Each dataset root folder is expected to contain:

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
  fno_aux/ or deeponet_aux/
    flat_ground.h5          # baseline p0 / p_flat for residual learning (if enabled)
```

The exact HDF5 keys can differ by dataset, but the scripts expect at least:
- coordinate grids `x_grid`, `z_grid`
- complex pressure channels `p_re`, `p_im`
- wind profile info for wind/joint datasets
- terrain profile info for terrain/joint datasets

---

## Quickstart

### Wind-only FNO (direct p)
```bash
python train_fno_wind_coords.py --data-root data/wind_sobol_complex_n_1000 --iid --device cuda
```

### Terrain-only FNO (η + mask; direct p by default)
```bash
python train_fno_terrain_eta_mask_phys.py --data-root data/terrain_sobol_complex_n_1000 --iid --device cuda
```

### Joint FNO (wind + terrain; residual Δp by default)
```bash
python train_fno_joint_wind_terrain_eta_mask.py --data-root data/wind_terrain_sobol_complex_perm_n_1000 --iid --device cuda
```

To force **direct p** in the joint script:
```bash
python train_fno_joint_wind_terrain_eta_mask.py --data-root data/wind_terrain_sobol_complex_perm_n_1000 --iid --device cuda --direct
```

---

## Outputs

Scripts write (by default):
- `checkpoints/<run_tag>.pt` (best)
- `checkpoints/<run_tag>__last.pt` (last)
- `runs/<run_tag>/metrics.csv`
- `evals/<run_tag>__<split>.json`

