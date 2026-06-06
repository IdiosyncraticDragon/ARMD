"""Linear-regression baseline for the Stock forecasting task, for comparison with ARMD.

The point of this script is an *apples-to-apples* comparison: it reuses the exact
same `CustomDataset` windows, z-score normalization, and train/test split that
`main.py` feeds to ARMD, and it computes MSE/MAE the same way (flattened, in
standardized space). The only thing that changes is the forecaster — instead of
the auto-regressive moving diffusion model, we fit a plain least-squares linear
map from the context window to the future window.

Task setup (identical to ARMD on Stock):
    - window = 192 = seq_length(96) context + 96 future
    - each sample: x[:96] is the observed context, x[96:] is the target future
    - features = 6 stock columns, standardized with StandardScaler (z-score)

The linear model maps the flattened context (96*6 = 576 dims) to the flattened
future (576 dims) via multi-output ordinary least squares (and, for reference, a
ridge-regularized variant). We also report a trivial "persistence" baseline
(repeat the last observed value) to contextualize the numbers.

Usage:
    python baselines/linear_regression_stock.py
    python baselines/linear_regression_stock.py --config_path ./Config/stock.yaml
    python baselines/linear_regression_stock.py --run_armd   # also train+eval ARMD live

Run from the repository root so that relative data paths resolve.
"""

import os
import sys
import copy
import random
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Allow running as `python baselines/linear_regression_stock.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

from Utils.io_utils import load_yaml_config, instantiate_from_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_split(dataset_cfg, save_dir, period, predict_length=None):
    """Instantiate a CustomDataset exactly as main.py would and return its windows.

    Returns an array of shape [num_windows, window, feature_size] in z-score space.
    """
    cfg = copy.deepcopy(dataset_cfg)
    cfg["params"]["output_dir"] = save_dir
    cfg["params"]["period"] = period
    if predict_length is not None:
        cfg["params"]["predict_length"] = predict_length
    dataset = instantiate_from_config(cfg)
    return np.asarray(dataset.samples), dataset


def split_context_future(samples, seq_len):
    """[N, 2*seq_len, F] -> (X flat [N, seq_len*F], Y flat [N, seq_len*F], future [N, seq_len, F])."""
    context = samples[:, :seq_len, :]
    future = samples[:, seq_len:, :]
    n = samples.shape[0]
    return context.reshape(n, -1), future.reshape(n, -1), future


def revin_stats(samples, seq_len, eps=1e-5):
    """Per-window lookback mean/std, mirroring ARMD's RevIN (stats from x[:, :seq_len, :])."""
    ctx = samples[:, :seq_len, :]
    mu = ctx.mean(axis=1, keepdims=True)
    sigma = np.sqrt(ctx.var(axis=1, keepdims=True) + eps)
    return mu, sigma


def fit_eval_linear(regressor, train_samples, test_samples, seq_len, revin):
    """Fit context->future, score MSE/MAE in outer-scaler space. With revin=True,
    fit/predict in per-window normalized space and de-normalize like ARMD does."""
    _, _, future_te = split_context_future(test_samples, seq_len)
    if revin:
        tr = (train_samples - revin_stats(train_samples, seq_len)[0]) / revin_stats(train_samples, seq_len)[1]
        mu_te, sig_te = revin_stats(test_samples, seq_len)
        te = (test_samples - mu_te) / sig_te
        ntr, nte, feat = tr.shape[0], te.shape[0], te.shape[-1]
        regressor.fit(tr[:, :seq_len, :].reshape(ntr, -1), tr[:, seq_len:, :].reshape(ntr, -1))
        pred = regressor.predict(te[:, :seq_len, :].reshape(nte, -1)).reshape(nte, seq_len, feat)
        pred = pred * sig_te + mu_te   # de-normalize to outer-scaler space
        return evaluate(pred, future_te)
    Xtr, Ytr, _ = split_context_future(train_samples, seq_len)
    Xte, Yte, _ = split_context_future(test_samples, seq_len)
    regressor.fit(Xtr, Ytr)
    return evaluate(regressor.predict(Xte), Yte)


def evaluate(pred_flat, true_flat):
    """Flattened MSE/MAE in standardized space — identical to main.py's metric call."""
    mse = mean_squared_error(pred_flat.reshape(-1), true_flat.reshape(-1))
    mae = mean_absolute_error(pred_flat.reshape(-1), true_flat.reshape(-1))
    return float(mse), float(mae)


def run_armd_live(configs, config_path, save_dir, gpu, seq_len):
    """Train + evaluate ARMD with the same pipeline as main.py, returning (mse, mae)."""
    import torch
    from engine.solver import Trainer
    from Data.build_dataloader import build_dataloader, build_dataloader_cond

    class _Args:
        pass

    args = _Args()
    args.config_path = config_path
    args.save_dir = save_dir
    args.gpu = gpu
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(configs["model"]).to(device)
    model.fast_sampling = True

    dataloader_info = build_dataloader(configs, args)
    trainer = Trainer(
        config=configs, args=args, model=model,
        dataloader={"dataloader": dataloader_info["dataloader"]},
    )
    trainer.train()

    args.mode = "predict"
    args.pred_len = seq_len
    test_info = build_dataloader_cond(configs, args)
    feat_num = test_info["dataset"].samples.shape[-1]

    mse_runs, mae_runs = [], []
    for run in range(10):
        torch.manual_seed(2023 + run)
        np.random.seed(2023 + run)
        random.seed(2023 + run)
        sample, real_ = trainer.sample_forecast(
            test_info["dataloader"], shape=[seq_len, feat_num]
        )
        mse_runs.append(mean_squared_error(sample.reshape(-1), real_.reshape(-1)))
        mae_runs.append(mean_absolute_error(sample.reshape(-1), real_.reshape(-1)))
    return float(np.mean(mse_runs)), float(np.mean(mae_runs))


def main():
    parser = argparse.ArgumentParser(
        description="Linear-regression baseline for Stock forecasting (compared with ARMD)."
    )
    parser.add_argument("--config_path", type=str, default="./Config/stock.yaml",
                        help="Config whose dataloader windows/split define the task.")
    parser.add_argument("--save_dir", type=str, default="./forecasting_exp",
                        help="Where CustomDataset writes its .npy artifacts.")
    parser.add_argument("--ridge_alpha", type=float, default=1.0,
                        help="L2 strength for the ridge variant.")
    parser.add_argument("--run_armd", action="store_true",
                        help="Also train+evaluate ARMD live for a head-to-head number.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for --run_armd.")
    args = parser.parse_args()

    set_seed(2023)
    configs = load_yaml_config(args.config_path)
    os.makedirs(args.save_dir, exist_ok=True)

    seq_len = int(configs["model"]["params"]["seq_length"])
    train_cfg = configs["dataloader"]["train_dataset"]
    test_cfg = configs["dataloader"]["test_dataset"]

    # Same windows ARMD trains on, and same test windows ARMD is scored on.
    train_samples, _ = build_split(train_cfg, args.save_dir, period="train")
    test_samples, _ = build_split(test_cfg, args.save_dir, period="test", predict_length=seq_len)

    _, _, future_te = split_context_future(test_samples, seq_len)
    feat = test_samples.shape[-1]
    # Match ARMD's normalization regime so the comparison stays apples-to-apples.
    revin = bool(configs["model"]["params"].get("use_revin", False))

    print("=" * 64)
    print("Linear-regression baseline vs ARMD on Stock")
    print("=" * 64)
    print(f"config            : {args.config_path}")
    print(f"seq_length (ctx)  : {seq_len}  ->  predict {seq_len} steps, {feat} features")
    print(f"train windows     : {train_samples.shape}")
    print(f"test  windows     : {test_samples.shape}")
    print(f"metric space      : z-score (StandardScaler), flattened MSE/MAE")
    print(f"per-window RevIN  : {revin}  (matches model.use_revin)")
    print("-" * 64)

    results = []

    # 1) Persistence: repeat the last observed context value across the horizon.
    #    (RevIN-invariant: normalizing then copying the last value then de-normalizing
    #    returns the same forecast, so it is computed once on raw windows.)
    last = test_samples[:, seq_len - 1:seq_len, :]  # [N, 1, F]
    persist = np.repeat(last, seq_len, axis=1)
    results.append(("Persistence (last value)", *evaluate(persist, future_te)))

    # 2) Ordinary least squares: context -> future.
    results.append(("Linear regression (OLS)",
                    *fit_eval_linear(LinearRegression(), train_samples, test_samples, seq_len, revin)))

    # 3) Ridge-regularized linear regression (often more stable for 576->576).
    results.append((f"Linear regression (Ridge a={args.ridge_alpha:g})",
                    *fit_eval_linear(Ridge(alpha=args.ridge_alpha), train_samples, test_samples, seq_len, revin)))

    # 4) ARMD — paper reference (different split, see caveat), and optionally a live run.
    cfg_l = args.config_path.replace("\\", "/").lower()
    if "stock" in cfg_l:
        results.append(("ARMD (paper Table 1, 70/10/20)", 0.235, 0.269))
    armd_live = None
    if args.run_armd:
        armd_live = run_armd_live(configs, args.config_path, args.save_dir,
                                  args.gpu, seq_len)
        results.append(("ARMD (this run, same split)", *armd_live))

    print(f"{'model':<34}{'MSE':>12}{'MAE':>12}")
    print("-" * 64)
    for name, mse, mae in results:
        print(f"{name:<34}{mse:>12.4f}{mae:>12.4f}")
    print("=" * 64)
    chrono = bool(test_cfg["params"].get("three_split", False))
    print("Lower is better.")
    if chrono:
        print(" * Chronological 70/10/20 split (leakage-free): the OLS/Ridge and ARMD")
        print("   rows are a true head-to-head on identical windows. Persistence is a")
        print("   strong baseline because Stock is close to a random walk.")
    else:
        print(" * The paper's 0.235/0.269 use a chronological 70/10/20 split. This")
        print("   config's default split (train=first 80%, test=last 80% of windows)")
        print("   overlaps heavily, so it is NOT comparable to the paper row -- only")
        print("   the 'same split' rows are a true head-to-head. A 576->576 linear map")
        print("   exploits that overlap, so use a three_split config for fairness.")


if __name__ == "__main__":
    main()
