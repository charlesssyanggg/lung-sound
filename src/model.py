"""
model.py — ResNet18-based Lung Sound Classifier

Modified from standard ResNet18:
  1. Input layer: 3-channel → 1-channel (grayscale Mel spectrogram)
  2. Classification head: Dropout → Linear(512,256) → ReLU → Linear(256,4)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


class LungSoundClassifier(nn.Module):
    """
    ResNet18 backbone adapted for single-channel Mel spectrogram input.

    Args:
        num_classes: Number of output classes (default: 4).
        pretrained:  If True, load ImageNet weights for backbone.
        dropout:     Tuple of (p1, p2) for two Dropout layers.

    Input:  (B, 1, 128, 128)  — batch × 1-channel × mel × time
    Output: (B, num_classes)  — raw logits
    """

    def __init__(self,
                 num_classes: int = 4,
                 pretrained: bool = False,
                 dropout: tuple = (0.4, 0.2)):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # ── Modification 1: 3-channel → 1-channel input ──────────
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        # If pretrained, average the 3 input channels → 1
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    self.backbone.conv1.weight.mean(dim=1, keepdim=True)
                )

        # ── Modification 2: Deeper classification head ────────────
        feat_dim = self.backbone.fc.in_features  # 512 for ResNet18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout[0]),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout[1]),
            nn.Linear(256, num_classes),
        )

        self._init_head()

    def _init_head(self):
        """Initialize classification head weights."""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_cam_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM."""
        return self.backbone.layer4[-1]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> LungSoundClassifier:
    """Build model from config dict."""
    return LungSoundClassifier(
        num_classes=cfg.get("num_classes", 4),
        pretrained=False,
        dropout=tuple(cfg.get("dropout", [0.4, 0.2])),
    )


if __name__ == "__main__":
    model = LungSoundClassifier()
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Grad-CAM target layer: {model.get_cam_layer()}")
