"""
train.py — Training Loop

Supports:
  - Configurable loss function (CE or Focal)
  - Cosine Annealing LR schedule
  - Gradient clipping
  - Best-checkpoint saving
  - TensorBoard-compatible loss/metric logging
"""

import os
import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm

from src.evaluate import evaluate, print_metrics


def train_one_epoch(model: nn.Module,
                    loader: torch.utils.data.DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: str,
                    grad_clip: float = 1.0) -> tuple:
    """
    Run one training epoch.

    Returns:
        (avg_loss, accuracy) both as Python floats
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, y in loader:
        x = x.to(device)
        y = (y.to(device) if isinstance(y, torch.Tensor)
             else torch.tensor(y, device=device))

        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total


def run_training(model: nn.Module,
                 train_loader,
                 test_loader,
                 criterion: nn.Module,
                 cfg: dict,
                 run_name: str = "experiment",
                 save_dir: str = "outputs") -> tuple:
    """
    Full training loop with evaluation and checkpoint saving.

    Args:
        model:        Model to train.
        train_loader: Training DataLoader.
        test_loader:  Validation/test DataLoader.
        criterion:    Loss function.
        cfg:          Config dict (train section).
        run_name:     Name for saved files.
        save_dir:     Directory for checkpoints and logs.

    Returns:
        (history, best_metrics, trained_model)
    """
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    epochs       = cfg.get("epochs", 40)
    lr           = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 1e-4)
    grad_clip    = cfg.get("grad_clip", 1.0)
    log_interval = cfg.get("log_interval", 5)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=cfg.get("eta_min", 1e-5)
    )

    history = {
        "train_loss": [],
        "train_acc":  [],
        "val_score":  [],
        "lr":         []
    }
    best_score = 0.0
    best_state = None
    best_metrics = {}

    print(f"\n{'═'*60}")
    print(f"  Run  : {run_name}")
    print(f"  Device: {device}  |  Epochs: {epochs}  |  LR: {lr}")
    print(f"  Loss : {criterion.__class__.__name__}")
    print(f"{'═'*60}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        metrics = evaluate(model, test_loader, device)
        score   = metrics["icbhi_score"]

        history["train_loss"].append(round(loss, 5))
        history["train_acc"].append(round(acc * 100, 2))
        history["val_score"].append(score)
        history["lr"].append(round(current_lr, 6))

        # Save best checkpoint
        if score > best_score:
            best_score   = score
            best_metrics = metrics
            best_state   = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            ckpt_path = os.path.join(save_dir, f"{run_name}_best.pth")
            torch.save({
                "epoch":        epoch,
                "model_state":  best_state,
                "icbhi_score":  best_score,
                "config":       cfg,
            }, ckpt_path)

        if epoch % log_interval == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Acc: {acc*100:.1f}% | "
                  f"ICBHI: {score:.2f}% | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.0f}s")

    # Load best weights for final evaluation
    model.load_state_dict(best_state)

    # Save training history
    hist_path = os.path.join(save_dir, f"{run_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✓ Best ICBHI Score : {best_score:.2f}%")
    print_metrics(best_metrics, title=f"Best Results — {run_name}")

    return history, best_metrics, model
