"""
models.py — Model architectures for HAM10000 skin-lesion classification.

Three models are provided:
  • CustomCNN        — lightweight baseline built from scratch
  • build_resnet50   — ResNet-50 with last block unfrozen (fine-tuning)
  • build_efficientnet_b3 — EfficientNet-B3 with last 3 feature blocks unfrozen
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 7


# ── Custom CNN ─────────────────────────────────────────────────────────────────

class CustomCNN(nn.Module):
    """Three-block convolutional baseline.

    Input  : (B, 3, 300, 300)
    Output : (B, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — 300 → 150
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2 — 150 → 75
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3 — 75 → 1×1 (adaptive pool)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # 128 × 1 × 1 = 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── ResNet-50 ─────────────────────────────────────────────────────────────────

def build_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ResNet-50 pre-trained on ImageNet.

    Strategy:
      - Freeze the full backbone.
      - Unfreeze ``layer4`` (the final residual block) for fine-tuning.
      - Replace the classification head with a two-layer MLP + Dropout.

    Returns:
        torch.nn.Module ready for training.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet50(weights=weights)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last residual block
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace head
    num_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )

    _print_param_summary(model, "ResNet-50")
    return model


# ── EfficientNet-B3 ───────────────────────────────────────────────────────────

def build_efficientnet_b3(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B3 pre-trained on ImageNet.

    Strategy:
      - Freeze the full backbone.
      - Unfreeze the last 3 feature blocks for fine-tuning.
      - Replace the classification head with a two-layer MLP + Dropout.

    Returns:
        torch.nn.Module ready for training.
    """
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.efficientnet_b3(weights=weights)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 feature blocks
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # Replace head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )

    _print_param_summary(model, "EfficientNet-B3")
    return model


# ── Utilities ──────────────────────────────────────────────────────────────────

def _print_param_summary(model: nn.Module, name: str) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Trainable: {trainable:,} / Total: {total:,}")


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function — returns the requested model by name.

    Args:
        model_name: One of ``"cnn"``, ``"resnet50"``, ``"efficientnet"``.
        **kwargs  : Forwarded to the model constructor.

    Returns:
        Instantiated ``torch.nn.Module``.
    """
    model_name = model_name.lower()
    registry = {
        "cnn":          CustomCNN,
        "resnet50":     build_resnet50,
        "efficientnet": build_efficientnet_b3,
    }
    if model_name not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(registry.keys())}"
        )
    return registry[model_name](**kwargs)
