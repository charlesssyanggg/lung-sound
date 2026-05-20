"""
focal_loss.py — Focal Loss (Innovation ①)

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    https://arxiv.org/abs/1708.02002

Why Focal Loss for lung sound classification?
    ICBHI 2017 class distribution:
        Normal:  3642 (52.8%)  ← majority
        Crackle: 1864 (27.0%)
        Wheeze:   886 (12.8%)  ← minority, clinically critical
        Both:     506 ( 7.3%)  ← minority

    Standard CrossEntropy optimizes for the majority class, resulting in
    very low Wheeze recall (~35%). Focal Loss down-weights easy (majority)
    samples via (1 - pt)^gamma, forcing the model to focus on hard
    (minority) samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional per-class alpha weighting.

    Formula:
        FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    Args:
        alpha (list or None):
            Per-class weight list of length num_classes.
            Set higher values for minority classes.
            If None, all classes are weighted equally.
            Recommended for ICBHI: [0.15, 0.28, 0.38, 0.19]
              (inverse-frequency, Wheeze gets highest weight)

        gamma (float):
            Focusing parameter. Higher = more focus on hard samples.
            gamma=0 reduces to standard weighted CrossEntropy.
            gamma=2 is the value recommended in the original paper.

        reduction (str):
            'mean' | 'sum' | 'none'

    Example:
        >>> criterion = FocalLoss(
        ...     alpha=[0.15, 0.28, 0.38, 0.19],  # ICBHI weights
        ...     gamma=2.0
        ... )
        >>> loss = criterion(logits, labels)
    """

    def __init__(self,
                 alpha: Optional[list] = None,
                 gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer(
                "alpha", torch.tensor(alpha, dtype=torch.float32)
            )
        else:
            self.alpha = None

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  (B, C) raw logits
            targets: (B,)   integer class labels in [0, C)

        Returns:
            Scalar loss (if reduction='mean') or tensor.
        """
        # Step 1: Standard cross-entropy loss (per sample, no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Step 2: Compute pt = exp(-CE) = probability of correct class
        pt = torch.exp(-ce_loss)

        # Step 3: Apply focal modulation factor
        focal_weight = (1.0 - pt) ** self.gamma

        # Step 4: Apply per-class alpha weight
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Step 5: Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss  # "none"

    def extra_repr(self) -> str:
        return (f"gamma={self.gamma}, "
                f"alpha={self.alpha.tolist() if self.alpha is not None else None}, "
                f"reduction='{self.reduction}'")


def build_criterion(cfg: dict, device: str = "cpu") -> nn.Module:
    """
    Build loss function from config dict.

    Args:
        cfg: config['focal_loss'] dict with keys: gamma, alpha
        device: torch device string

    Returns:
        FocalLoss instance
    """
    return FocalLoss(
        alpha=cfg.get("alpha", None),
        gamma=cfg.get("gamma", 2.0),
        reduction="mean",
    )


if __name__ == "__main__":
    # ── Sanity check ───────────────────────────────────────
    torch.manual_seed(0)
    B, C = 8, 4
    logits  = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))

    ce   = nn.CrossEntropyLoss()(logits, targets)
    fl   = FocalLoss(alpha=[0.15, 0.28, 0.38, 0.19], gamma=2.0)(logits, targets)
    fl_0 = FocalLoss(alpha=None, gamma=0)(logits, targets)   # Should ≈ CE

    print(f"CrossEntropy:        {ce.item():.4f}")
    print(f"FocalLoss (gamma=2): {fl.item():.4f}")
    print(f"FocalLoss (gamma=0): {fl_0.item():.4f}  ← should ≈ CE={ce.item():.4f}")
