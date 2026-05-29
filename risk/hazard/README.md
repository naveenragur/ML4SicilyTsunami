# Hazard Preparation Folder

This folder is organized so the runnable Slurm entrypoints stay at the top
level, while generated OpenQuake input arrays and helper code live separately.

## Layout

- `run.sbatch`: main Slurm job for preparing filtered hazard arrays and building
  OpenQuake HDF5 hazard inputs.
- `run.2upload`: upload/compression helper script.
- `npy_inputs/`: filtered NumPy triplets (`pred_d`, `event_id`, `site_id`).
- `oq_inputs/`: generated OpenQuake HDF5 GMF inputs.
- `scripts/`: Python helper scripts for preparing filtered arrays and writing
  OpenQuake HDF5 files.
- `notebooks/`: exploratory notebooks.
- `log/`: Slurm stdout/stderr logs.

## Main Workflow

From this directory:

```bash
sbatch run.sbatch
```

The job writes filtered `.npy` triplets to `npy_inputs/`, then reads those
arrays to create OpenQuake HDF5 inputs in `oq_inputs/`.
