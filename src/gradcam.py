"""
gradcam.py — Grad-CAM Explainability Visualization (Innovation ③)

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
    https://arxiv.org/abs/1610.02391

Usage:
    python src/gradcam.py --model_path outputs/focal_best.pth
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.model   import LungSoundClassifier
from src.dataset import ICBHIDataset, get_patient_split

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


# ══════════════════════════════════════════════════════════
# Grad-CAM Implementation
# ══════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Registers forward/backward hooks on a target layer to capture
    feature maps and gradients, then computes a spatial attention map.

    Args:
        model:        Trained model.
        target_layer: Conv layer to visualize (model.get_cam_layer()).
    """

    def __init__(self, model: torch.nn.Module,
                 target_layer: torch.nn.Module):
        self.model   = model
        self._feats  = None
        self._grads  = None

        self._fh = target_layer.register_forward_hook(self._fwd_hook)
        self._bh = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._feats = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def __call__(self, x: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        """
        Compute Grad-CAM for a single input tensor.

        Args:
            x:         Input tensor (1, 1, H, W).
            class_idx: Target class. If None, uses predicted class.

        Returns:
            cam: (H, W) numpy array, values in [0, 1].
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pooling of gradients → channel weights
        weights = self._grads[0].mean(dim=(1, 2))    # (C,)
        cam     = (weights[:, None, None] * self._feats[0]).sum(0)  # (H, W)
        cam     = F.relu(cam)
        cam     = cam / (cam.max() + 1e-8)

        # Upsample to input size
        cam_up = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        return cam_up.cpu().numpy(), class_idx

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()


# ══════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════

def find_one_per_class(dataset: torch.utils.data.Dataset) -> dict:
    """Find one representative sample per class."""
    samples = {i: None for i in range(len(CLASS_NAMES))}
    for i in range(len(dataset)):
        x, y = dataset[i]
        label = int(y)
        if samples[label] is None:
            samples[label] = (x, label)
        if all(v is not None for v in samples.values()):
            break
    return samples


def plot_gradcam(model: LungSoundClassifier,
                 samples: dict,
                 device: str,
                 save_path: str = "figures/04_gradcam.png"):
    """
    Generate a 3-row × 4-column Grad-CAM visualization figure.

    Row 0: Original Mel spectrogram
    Row 1: Grad-CAM heatmap
    Row 2: Overlay (spectrogram + heatmap)
    """
    cam_engine = GradCAM(model, model.get_cam_layer())
    model = model.to(device)

    fig, axes = plt.subplots(3, 4, figsize=(14, 9),
                              facecolor="white")

    for col, cls_idx in enumerate(range(len(CLASS_NAMES))):
        item = samples[cls_idx]
        if item is None:
            for row in range(3):
                axes[row, col].axis("off")
            continue

        x, label = item
        x_in = x.unsqueeze(0).to(device)

        cam, pred = cam_engine(x_in, class_idx=cls_idx)
        spec = x[0].cpu().numpy()

        # Normalize spectrogram to [0, 1] for display
        spec_n = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        cam_n  = (cam  - cam.min())  / (cam.max()  - cam.min()  + 1e-8)

        pred_correct = (pred == label)
        title_color  = "green" if pred_correct else "red"
        pred_str     = ("✓ " if pred_correct else "✗ ") + CLASS_NAMES[pred]

        # Row 0: Mel spectrogram
        axes[0, col].imshow(spec_n, aspect="auto", origin="lower",
                             cmap="magma")
        axes[0, col].set_title(
            f"{CLASS_NAMES[cls_idx]}\n(pred: {pred_str})",
            fontsize=10, fontweight="bold", color=title_color
        )

        # Row 1: Grad-CAM heatmap
        im = axes[1, col].imshow(cam_n, aspect="auto", origin="lower",
                                  cmap="jet")
        axes[1, col].set_title("Grad-CAM", fontsize=9)

        # Row 2: Overlay
        overlay = 0.45 * spec_n + 0.55 * cam_n
        axes[2, col].imshow(overlay, aspect="auto", origin="lower",
                             cmap="jet")
        axes[2, col].set_title("Overlay", fontsize=9)

        # Add acoustic annotation
        notes = {
            1: "Low-freq\nburst (Crackle)",
            2: "High-freq\nband (Wheeze)",
            3: "Low + High\nfreq (Both)",
        }
        if cls_idx in notes:
            axes[1, col].text(
                2, spec_n.shape[0] * 0.92, notes[cls_idx],
                color="white", fontsize=7.5, fontweight="bold",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="black", alpha=0.65)
            )

        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    axes[0, 0].set_ylabel("Mel Spectrogram\n(freq ↑)", fontsize=10)
    axes[1, 0].set_ylabel("Grad-CAM\n(red = high attention)", fontsize=10)
    axes[2, 0].set_ylabel("Overlay", fontsize=10)

    fig.suptitle(
        "Grad-CAM Explainability Analysis (Innovation ③)\n"
        "Model attention aligns with lung sound acoustic physics",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    cam_engine.remove_hooks()
    print(f"  [Figure] Saved: {save_path}")


# ── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grad-CAM Explainability (Innovation ③)"
    )
    parser.add_argument("--model_path", default="outputs/focal_best.pth")
    parser.add_argument("--config",     default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    ckpt  = torch.load(args.model_path, map_location=device)
    model = LungSoundClassifier(num_classes=4)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    # Load dataset
    data_cfg = cfg["data"]
    _, test_ids = get_patient_split(data_cfg["icbhi_root"])
    test_ds = ICBHIDataset(
        data_cfg["icbhi_root"], test_ids, data_cfg, augment=False
    )

    # Find one sample per class
    samples = find_one_per_class(test_ds)
    print(f"Found samples: { {CLASS_NAMES[k]: v is not None for k, v in samples.items()} }")

    # Generate Grad-CAM figure
    plot_gradcam(model, samples, device,
                 save_path="figures/04_gradcam.png")
