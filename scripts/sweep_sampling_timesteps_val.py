"""
Sweep sampling_timesteps on the validation split (paper supplemental: choose from {1..12}).
Requires a trained checkpoint under results_folder + _{seq_length}.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Data.build_dataloader import build_dataloader
from engine.solver import Trainer
from Utils.io_utils import instantiate_from_config, load_yaml_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--checkpoint_milestone", type=int, default=1)
    p.add_argument("--ts_min", type=int, default=1)
    p.add_argument("--ts_max", type=int, default=12)
    return p.parse_args()


def build_val_loader(config: dict, args, seq_len: int):
    """Val split with predict-style masking (same as test)."""
    batch_size = config["dataloader"]["sample_size"]
    cfg = copy.deepcopy(config)
    cfg["dataloader"]["test_dataset"]["params"]["output_dir"] = args.save_dir
    cfg["dataloader"]["test_dataset"]["params"]["period"] = "val"
    cfg["dataloader"]["test_dataset"]["params"]["predict_length"] = seq_len
    test_dataset = instantiate_from_config(cfg["dataloader"]["test_dataset"])
    dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return dl, test_dataset


def main():
    args = parse_args()
    set_seed(2023)
    seq_len = 96
    configs = load_yaml_config(args.config_path)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(configs["model"]).to(device)
    model.fast_sampling = True

    class Args:
        config_path = args.config_path
        save_dir = args.save_dir
        gpu = args.gpu
        name = "sweep"

    tr_args = Args()
    dataloader_info = build_dataloader(copy.deepcopy(configs), tr_args)
    trainer = Trainer(
        config=copy.deepcopy(configs),
        args=tr_args,
        model=model,
        dataloader={"dataloader": dataloader_info["dataloader"]},
    )
    trainer.load(args.checkpoint_milestone, verbose=False)

    val_loader, _ = build_val_loader(load_yaml_config(args.config_path), tr_args, seq_len)
    feat_num = val_loader.dataset.samples.shape[-1]

    results = []
    for ts in range(args.ts_min, args.ts_max + 1):
        m = trainer.ema.ema_model
        m.sampling_timesteps = ts
        m.fast_sampling = ts < m.num_timesteps
        set_seed(2023)
        pred, real = trainer.sample_forecast(val_loader, shape=[seq_len, feat_num])
        mse = mean_squared_error(pred.reshape(-1), real.reshape(-1))
        mae = mean_absolute_error(pred.reshape(-1), real.reshape(-1))
        results.append({"sampling_timesteps": ts, "val_mse": mse, "val_mae": mae})
        print(ts, "val_mae", mae, "val_mse", mse)

    best = min(results, key=lambda r: r["val_mae"])
    print("best_by_val_mae", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
