# Auto-Regressive Moving Diffusion Models for Time Series Forecasting

This is the official repo for "Auto-Regressive Moving Diffusion Models for Time Series Forecasting".

## Requirements

1. Install Python >= 3.10.

2. Required dependencies can be installed by: 
   
   ```
   pip install -r requirements.txt
   ```

**Tutorial notebook (minimal env with [uv](https://docs.astral.sh/uv/)):** see [`tutorials/armd_stock_tutorial.ipynb`](tutorials/armd_stock_tutorial.ipynb).

First install deps from [`pyproject.toml`](pyproject.toml): `uv sync`

Then start Jupyter with **one** of the following (there is **no** `notebook` executable: `uv run notebook` fails with “program not found”):

```
uv run jupyter notebook tutorials/armd_stock_tutorial.ipynb
```

```
uv run python -m notebook tutorials/armd_stock_tutorial.ipynb
```

```
uv run jupyter-notebook tutorials/armd_stock_tutorial.ipynb
```

Headless execution (from repository root; set `ARMD_REPO` if cwd is wrong). Use **`--no-sync`** so `uv run` does not reinstall `torch` on every invocation:

```
uv run --no-sync python -m jupyter nbconvert --execute tutorials/armd_stock_tutorial.ipynb --inplace
```

For full parity with the original experiments (GluonTS, MuJoCo, etc.), install `requirements.txt` in a separate environment.

### Tutorial: GPU-enabled PyTorch (CUDA 12.4)

[`pyproject.toml`](pyproject.toml) installs **`torch` from PyTorch’s CUDA 12.4 wheel index** (`[[tool.uv.index]]` + `[tool.uv.sources]`), not the CPU-only wheel from PyPI.

```bash
uv lock
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect a version suffix like `+cu124` and `True` when your NVIDIA driver is working. In Jupyter, pick the kernel that uses **`.venv`** (same interpreter as above).

**CPU-only:** comment out or remove the `[[tool.uv.index]]` block and `[tool.uv.sources]` under `[tool.uv.sources]` in `pyproject.toml`, then run `uv lock` and `uv sync` again.

**Broken `torch` after interrupted install** (e.g. `ImportError: FakeWork`): close Jupyter/IDE using the venv, then run `uv sync --reinstall-package torch`.

## Dataset Preparation

The datasets (ETT, Solar Energy and Exchange) can be obtain from (https://github.com/thuml/iTransformer ). **Stock** (and related archives from Diffusion-TS) can be obtained from (https://github.com/Y-debug-sys/Diffusion-TS ): download **`dataset.zip`** from the Google Drive link in that README, unzip, and copy **`stock_data.csv`** into `./Data/datasets/stock_data.csv`.

The Diffusion-TS **stock** file has **six** numeric columns (`Open`, `High`, `Low`, `Close`, `Adj_Close`, `Volume`) and **no** leading date column. Use [`Config/stock.yaml`](Config/stock.yaml) as provided (`feature_size: 6`, `name: stock`). Do **not** use `name: etth` for this CSV—that mode drops the first column (intended for date-stamped ETTh data) and would incorrectly remove `Open`.

### Training & Sampling

For training & Sampling, you can run:

~~~bash
python main.py --config_path ./Config/etth1.yaml
~~~

**Note:** We provide the corresponding `.yaml` files under the folder `./Config` where all possible options can be altered. You may need to change some hyper-parameters in the model for different forecasting scenarios.


## Acknowledgement

We appreciate the following github repos for their valuable codes:

https://github.com/lucidrains/denoising-diffusion-pytorch

https://github.com/Y-debug-sys/Diffusion-TS

https://github.com/thuml/iTransformer

https://github.com/zalandoresearch/pytorch-ts

https://github.com/Hundredl/MG-TSD

https://github.com/paddlepaddle/paddlespatial

https://github.com/amazon-science/unconditional-time-series-diffusion
