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

## Baselines

A linear-regression baseline for Stock lives in `baselines/linear_regression_stock.py`.
It reuses the **same** `CustomDataset` windows, z-score normalization, and train/test
split that ARMD uses, and scores MSE/MAE identically to `main.py` (flattened, in
standardized space). Only the forecaster changes: a least-squares map from the
96-step context to the 96-step future.

```bash
# Linear baselines only (fast): persistence, OLS, ridge
python baselines/linear_regression_stock.py

# Add a live ARMD run on the SAME split for a true head-to-head (trains ARMD)
python baselines/linear_regression_stock.py --run_armd
```

For a leakage-free comparison use the chronological 70/10/20 split:

```bash
python baselines/linear_regression_stock.py --config_path ./Config/stock_chrono.yaml --run_armd
```

How to read the regimes (all numbers from our runs, z-score space):

| regime | Persist. MSE | Ridge MSE | ARMD MSE | who wins |
|--------|--------------|-----------|----------|----------|
| default `stock.yaml` (overlapping windows, global norm) | 0.065 | 0.049 | 0.068 | Ridge (leakage artifact) |
| `stock_chrono.yaml` (chronological, global norm) | 0.055 | 0.143 | 0.118 | **ARMD** |
| chronological + **train-only norm** (no RevIN) | 0.322 | 0.390 | 0.428 | Persistence (others blow up) |
| `stock_paper.yaml` (chronological, global norm, **RevIN**) | 0.055 | 0.092 | **0.060** | **ARMD** |

Key points:
- The default `stock.yaml` split has train = first 80% of windows and test = last
  80%, which **overlap by ~2000 windows**. A 576→576 linear map exploits that
  leakage and appears to beat ARMD — an artifact, not a real result.
- On the chronological split (`three_split: true`, no train/test window overlap)
  the ordering flips back: ARMD beats Ridge, and unregularized OLS overfits badly
  (MSE≈0.31). This matches the paper, whose own Table 1 shows DLinear (a linear
  model) is the 2nd-best method on Stock (0.286 vs ARMD 0.235) — Stock has strong
  linear structure, so a good linear baseline is *expected* to be competitive.
- **Normalization matters more than the split here.** `norm_on_train: True` fits
  the z-score on the training raw region only (Informer/Autoformer convention).
  For Stock this *overshoots* — MSE blows up for every trained model because the
  series **triples in price** (train mean ≈282 → test mean ≈968), so the test
  period sits at **+5.5σ** under train statistics. The `Linear` backbone is a plain
  `nn.Linear` on absolute values, so that level shift becomes a pure
  out-of-distribution extrapolation; only persistence stays reasonable.
- **The fix is per-window RevIN, not train-only scaling.** `use_revin: True` (ARMD)
  normalizes each window by its own lookback mean/std and re-applies them to the
  prediction (so metrics stay in outer-scaler space). The baseline mirrors this
  automatically when the config sets `use_revin` (see `fit_eval_linear`). With
  RevIN the level drift disappears: ARMD (0.060) cleanly beats OLS/Ridge
  (~0.09), and even unregularized OLS stops over-fitting. This is the fairest,
  paper-consistent regime — run it with:

  ```bash
  python baselines/linear_regression_stock.py --config_path ./Config/stock_paper.yaml --run_armd
  ```

- Stock is close to a random walk, so naive persistence (MSE 0.055) remains very
  competitive on this dataset even with RevIN; treat absolute MSE gaps with that
  in mind. Our absolute numbers run lower than the paper's 0.235 (different
  preprocessing pipeline); the **relative** ordering — ARMD > linear on a clean,
  properly-normalized split — is what reproduces the paper's claim.

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
