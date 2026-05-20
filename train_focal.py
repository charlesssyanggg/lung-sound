"""
train_focal.py — Run Experiment B: Focal Loss (Innovation ①)

Usage:
    python train_focal.py
    python train_focal.py --config configs/config.yaml --save_dir outputs
"""

import argparse
import torch
import yaml

from src.dataset    import get_icbhi_loaders
from src.model      import build_model
from src.focal_loss import build_criterion
from src.train      import run_training
from src.evaluate   import print_metrics


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

    # Loss: Focal Loss (Innovation ①)
    criterion = build_criterion(cfg["focal_loss"])
    print(f"[Loss] {criterion}")

    # Train
    history, metrics, model = run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        cfg=cfg["train"],
        run_name="focal_loss",
        save_dir=args.save_dir
    )

    print_metrics(metrics, title="FINAL — Focal Loss (Innovation ①)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--save_dir", default="outputs")
    main(parser.parse_args())
