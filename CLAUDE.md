# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARMD (Auto-Regressive Moving Diffusion) is a PyTorch-based time series forecasting model. The core idea: instead of adding Gaussian noise, the forward diffusion process applies an auto-regressive moving operation — shifting the observed window — so the model learns to denoise by predicting future values from past context.

The paper: [arxiv 2412.09328](https://arxiv.org/abs/2412.09328). Paper-aligned reference result: Stock MSE=0.235, MAE=0.269 (Table 1, z-score).

## Environment Setup

Two separate environments exist for different purposes:

**Tutorial env (minimal, GPU-enabled via uv):**
```bash
uv sync                          # installs from pyproject.toml with CUDA 12.4 torch
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Full experiment env (GluonTS, MuJoCo, etc.):**
```bash
pip install -r requirements.txt
```

To switch torch to CPU-only: comment out `[[tool.uv.index]]` and `[tool.uv.sources]` in `pyproject.toml`, then `uv lock && uv sync`.

Fix broken torch after interrupted install: `uv sync --reinstall-package torch`.

## Training & Evaluation

```bash
# Train and evaluate (train + sample, metrics averaged over 10 runs)
python main.py --config_path ./Config/stock.yaml

# Other datasets
python main.py --config_path ./Config/etth1.yaml
python main.py --config_path ./Config/ettm1.yaml
python main.py --config_path ./Config/exchange.yaml
```

`--save_dir` (default `./forecasting_exp`) and `--gpu` (default `0`) are optional flags.

## Tutorials

```bash
# Interactive Jupyter (uses project modules via imports)
uv run jupyter notebook tutorials/armd_stock_tutorial.ipynb

# Standalone self-contained notebook (no project imports)
uv run jupyter notebook tutorials/armd_standalone_full.ipynb

# Headless execution
uv run --no-sync python -m jupyter nbconvert --execute tutorials/armd_stock_tutorial.ipynb --inplace

# Regenerate the standalone notebook after modifying Models/ or engine/
python tutorials/generate_armd_standalone_notebook.py
```

Always run Jupyter from the **repository root** so that relative paths like `Data/datasets/` and `Config/` resolve correctly.

## Architecture

```
main.py                          # Entry point: train → sample → print MSE/MAE
engine/solver.py                 # Trainer class: Adam + EMA + LR scheduler, save/load checkpoints
Models/autoregressive_diffusion/
  armd.py                        # ARMD: diffusion wrapper, forward/reverse process, loss
  linear.py                      # Linear: learnable AR-moving backbone (the denoiser)
  model_utils.py                 # Shared utilities: positional encodings, LayerNorm, RevIN, etc.
Data/build_dataloader.py         # build_dataloader (train) / build_dataloader_cond (test/predict)
Utils/
  Data_utils/real_datasets.py    # CustomDataset: windowed sliding samples, z-score/[-1,1] scaling
  Data_utils/data_loader.py      # Dataset loaders for ETT, Exchange, Solar, fMRI formats
  io_utils.py                    # instantiate_from_config (registry pattern), load_yaml_config
  metric_utils.py                # MSE, MAE, and other evaluation metrics
Config/                          # Per-dataset YAML configs (model + solver + dataloader sections)
```

### Key design patterns

**Config-driven instantiation:** Every major class (model, dataset, LR scheduler) is referenced by its dotted Python path in YAML under `target:`, with constructor args under `params:`. `instantiate_from_config()` in `io_utils.py` does `importlib` lookup and calls the class.

**ARMD forward pass:** Input shape is `[batch, seq_length*2, feature_size]` (context + future concatenated). The `q_sample` "noising" step is an index-shift along the time axis (not Gaussian noise addition). The `Linear` backbone predicts the clean future from the shifted context; training loss is L1/L2 on predicted vs. true noise. At inference, `fast_sample` runs DDIM-style sampling with `sampling_timesteps=1` by default.

**Dataset naming gotcha:** `stock_data.csv` has 6 numeric columns with no leading date column. Use `name: stock` in the YAML. Using `name: etth` incorrectly drops the first column (intended to strip a date string from ETTh CSVs).

**Checkpoint folder:** Appends the model's `seq_length` to the `results_folder` path in the YAML. E.g., `results_folder: ./Checkpoints_stock` → saves to `./Checkpoints_stock_96/`.

## Data Preparation

- **ETT / Exchange / Solar:** download from [iTransformer repo](https://github.com/thuml/iTransformer), place in `./Data/datasets/`.
- **Stock:** download `dataset.zip` from [Diffusion-TS Google Drive](https://github.com/Y-debug-sys/Diffusion-TS), unzip, copy `stock_data.csv` → `./Data/datasets/stock_data.csv`.
- **fMRI:** `.mat` files already present in `./Data/datasets/fMRI/`.
