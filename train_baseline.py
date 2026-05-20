"""
train_baseline.py — Run Experiment A: Baseline (CrossEntropy Loss)

Usage:
    python train_baseline.py
    python train_baseline.py --config configs/config.yaml --save_dir outputs
"""

import argparse
import os
import torch
import torch.nn as nn
import yaml

from src.dataset   import get_icbhi_loaders
from src.model     import build_model
from src.train     import run_training
from src.evaluate  import print_metrics


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["train"]["seed"])

    # Data
    train_loader, test_loader = get_icbhi_loaders(
        cfg["data"]["icbhi_root"],
        cfg["data"],
        batch_size=cfg["train"]["batch_size"]
    )

    # Model
    model = build_model(cfg["model"])
    print(f"[Model] Parameters: {model.count_parameters():,}")

    # Loss: Standard CrossEntropy (Baseline)
    criterion = nn.CrossEntropyLoss()
    print(f"[Loss] CrossEntropyLoss (Baseline)")

    # Train
    history, metrics, model = run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        cfg=cfg["train"],
        run_name="baseline",
        save_dir=args.save_dir
    )

    print_metrics(metrics, title="FINAL — Baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--save_dir", default="outputs")
    main(parser.parse_args())
