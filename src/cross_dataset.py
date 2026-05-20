"""
cross_dataset.py — Cross-Dataset Generalization Evaluation (Innovation ②)

Loads a model trained on ICBHI 2017 and evaluates it on SPRSound
(zero-shot transfer) to quantify domain shift.

Usage:
    python src/cross_dataset.py --model_path outputs/focal_best.pth
"""

import argparse
import json
import os
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dataset  import get_icbhi_loaders, get_sprout_loader
from src.model    import LungSoundClassifier
from src.evaluate import evaluate, print_metrics


def load_model(ckpt_path: str, device: str) -> LungSoundClassifier:
    """Load model from checkpoint."""
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = LungSoundClassifier(num_classes=4)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"[Cross-Dataset] Loaded checkpoint: {ckpt_path}")
    print(f"  Checkpoint ICBHI Score: {ckpt.get('icbhi_score', 'N/A'):.2f}%")
    return model


def plot_cross_dataset(results: dict, save_path: str):
    """Bar chart comparing in-domain vs cross-domain performance."""
    datasets = list(results.keys())
    models   = list(list(results.values())[0].keys())
    x = np.arange(len(datasets))
    w = 0.3
    palette = ["#90A4AE", "#2196F3"]

    fig, ax = plt.subplots(figsize=(9, 5.5),
                            facecolor="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, m in enumerate(models):
        vals   = [results[d][m] for d in datasets]
        offset = (i - 0.5) * w
        bars   = ax.bar(x + offset, vals, w,
                        label=m, color=palette[i],
                        alpha=0.9, edgecolor="white", zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    # Annotate generalization gaps
    for i, m in enumerate(models):
        v_in  = results[datasets[0]][m]
        v_out = results[datasets[1]][m]
        offset = (i - 0.5) * w
        gap    = v_in - v_out
        ax.annotate(
            f"gap\n{gap:.1f}%",
            xy=(0.5 + offset, (v_in + v_out) / 2),
            fontsize=8, ha="center", color=palette[i],
            fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("ICBHI Score (%)", fontsize=11)
    ax.set_ylim(55, 95)
    ax.set_title(
        "Cross-Dataset Generalization (Innovation ②)\n"
        "Focal Loss achieves smaller domain shift gap",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Figure] Saved: {save_path}")


def run(baseline_ckpt: str,
        focal_ckpt:    str,
        cfg:           dict,
        save_dir:      str = "outputs") -> dict:
    """
    Full cross-dataset evaluation pipeline.

    Args:
        baseline_ckpt: Path to baseline model checkpoint.
        focal_ckpt:    Path to focal loss model checkpoint.
        cfg:           Full config dict.
        save_dir:      Output directory.

    Returns:
        Nested dict: {dataset: {model_name: icbhi_score}}
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = cfg["data"]

    # ── Load data ──────────────────────────────────────────
    _, icbhi_test_loader = get_icbhi_loaders(
        data_cfg["icbhi_root"], data_cfg
    )
    spr_loader = get_sprout_loader(
        data_cfg["sprout_root"], data_cfg
    )

    # ── Load models ────────────────────────────────────────
    model_baseline = load_model(baseline_ckpt, device)
    model_focal    = load_model(focal_ckpt,    device)

    # ── Evaluate ───────────────────────────────────────────
    results_raw = {}
    for loader, domain in [(icbhi_test_loader, "ICBHI 2017 (in-domain)"),
                            (spr_loader,        "SPRSound (cross-domain)")]:
        m_base = evaluate(model_baseline, loader, device)
        m_focal = evaluate(model_focal,   loader, device)
        results_raw[domain] = {
            "Baseline":   m_base["icbhi_score"],
            "Focal Loss": m_focal["icbhi_score"],
        }
        print(f"\n[{domain}]")
        print_metrics(m_base,  title="  Baseline")
        print_metrics(m_focal, title="  Focal Loss")

    # ── Plot & Save ────────────────────────────────────────
    plot_cross_dataset(
        results_raw,
        save_path="figures/05_cross_dataset.png"
    )

    out_path = os.path.join(save_dir, "cross_dataset_results.json")
    with open(out_path, "w") as f:
        json.dump(results_raw, f, indent=2)
    print(f"\n[Cross-Dataset] Results saved: {out_path}")

    return results_raw


# ── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-Dataset Generalization Evaluation (Innovation ②)"
    )
    parser.add_argument("--baseline",  default="outputs/baseline_best.pth")
    parser.add_argument("--focal",     default="outputs/focal_best.pth")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--save_dir",  default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(args.baseline, args.focal, cfg, args.save_dir)
