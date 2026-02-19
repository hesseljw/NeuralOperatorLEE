# NeuralOperatorLEE

Neural-operator surrogates for outdoor acoustics generated with a 2D Linearized Euler Equations (LEE) solver.
This repo contains training scripts for **DeepONet** and **Fourier Neural Operator (FNO)** models on wind-only, terrain-only, and joint wind+terrain cases.

## What’s inside

### Training scripts
- `train_deeponet_wind_gridtrunk.py`  
  DeepONet for **wind-only**: branch encodes wind-profile inputs, trunk is a grid/spectral (FNO-like) trunk, predicts complex pressure on the ROI.

- `train_deeponet_terrain_eta_fourier.py`  
  DeepONet for **terrain-only**: uses terrain parameterization (e.g., η = z − h(x)) and optional Fourier features, supports air-masked loss.

- `train_fno_wind_coords.py`  
  FNO for **wind-only**: grid-to-grid model with coordinate features, predicts complex pressure on the ROI.

- `train_fno_terrain_eta_mask_phys.py`  
  FNO for **terrain-only**: η + air-mask handling (and optional physics-style terms depending on your config), predicts complex pressure on the ROI.

- `train_fno_joint_wind_terrain_eta_mask.py`  
  FNO for **wind + terrain** jointly: includes η + air-mask and supports training on a visualization subset if enabled.

### Outputs
Each script writes a run folder (checkpoints + metrics) under a configurable output directory (see `USER CONFIG` in each file).

## Quickstart


###  Run a training script
```bash
python train_fno_wind_coords.py --data-root /path/to/data --device cuda
python train_fno_joint_wind_terrain_eta_mask.py --data-root /path/to/data --device cuda
```

All scripts also have a **`USER CONFIG`** section at the top (paths, batch size, epochs, model width/modes/layers, etc.).

## Data availability

The datasets used in the accompanying work are **available by request**.  
Please contact the author(s) to obtain access and details about the dataset format.

## Target choice: `p` vs `Δp`

All training scripts support predicting either:
- **direct** complex pressure `p(x,z)` (re/im), or
- **residual** `Δp(x,z) = p − p_baseline`, with reconstruction `p = p_baseline + Δp`.

This is controlled by a single boolean/CLI flag in each script (see the `USER CONFIG` block at the top).

## Notes

- Loss/metrics can be **masked to the air region** (`z ≥ h(x)`) for terrain/joint cases.
- Default hyperparameters are set to reasonable “paper baseline” values but are easy to change.

## Citation

If you use this code in academic work, please cite the associated paper/preprint (TBA). 
