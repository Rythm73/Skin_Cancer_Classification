"""
utils.py — Helper functions for metrics, plotting, and Grad-CAM.
"""

from __future__ import annotations

import os
import copy
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from .dataset import CLASS_NAMES, LABEL_NAMES

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CLASS_COLORS = [
    "#E74C3C", "#8E44AD", "#2980B9", "#27AE60",
    "#F39C12", "#16A085", "#2C3E50",
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    test_loader,
    model_name: str,
    device: torch.device,
    save_dir: Optional[str] = None,
) -> dict:
    """Run inference on *test_loader* and print a full metrics report.

    Args:
        model       : Trained PyTorch model (eval mode is set internally).
        test_loader : DataLoader for the test split.
        model_name  : Label used in printed output and saved figure filenames.
        device      : Torch device.
        save_dir    : If provided, confusion matrix PNG is saved here.

    Returns:
        dict with keys: accuracy, f1_weighted, f1_macro, precision,
        recall, roc_auc.
    """
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs  = inputs.to(device)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = torch.argmax(probs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_probs = np.array(y_probs)

    metrics = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "precision":   precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":      recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc":     roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"),
    }

    print(f"\n{'='*55}")
    print(f"  TEST RESULTS — {model_name}")
    print(f"{'='*55}")
    for name, val in metrics.items():
        print(f"  {name:<20}: {val:.4f}")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    _plot_confusion_matrix(y_true, y_pred, model_name, save_dir)
    return metrics


def _plot_confusion_matrix(y_true, y_pred, model_name, save_dir=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"{model_name} — Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(history, model_name: str, save_dir: Optional[str] = None):
    """Plot loss, accuracy, and F1 curves for a single training run."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(model_name, fontsize=14)

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history["val_f1"], color="green")
    axes[2].set_title("Val F1 Weighted")
    axes[2].grid(True)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"curves_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


def plot_model_comparison(histories: dict, save_dir: Optional[str] = None):
    """Overlay val curves for all models on the same axes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("All Models — Comparison", fontsize=14)

    for name, h in histories.items():
        axes[0].plot(h["val_loss"], label=name)
        axes[1].plot(h["val_acc"],  label=name)
        axes[2].plot(h["val_f1"],   label=name)

    titles = ["Val Loss — lower is better",
              "Val Accuracy — higher is better",
              "Val F1 Weighted — higher is better"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── ROC curves ────────────────────────────────────────────────────────────────

def get_probs_and_labels(model, loader, device):
    model.eval()
    y_true, y_probs = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs  = inputs.to(device)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            y_true.extend(labels.numpy())
            y_probs.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_probs)


def plot_roc_curves(
    models_dict: dict,
    test_loader,
    device: torch.device,
    save_dir: Optional[str] = None,
):
    """Plot per-class OvR ROC curves for each model side-by-side.

    Args:
        models_dict : ``{"Model Name": model, ...}``
    """
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle("ROC Curves — HAM10000 | One-vs-Rest per Class",
                 fontsize=15, fontweight="bold", y=1.02)

    auc_results = {}
    for ax, (model_name, model) in zip(axes, models_dict.items()):
        y_true, y_probs = get_probs_and_labels(model, test_loader, device)
        auc_results[model_name] = _plot_single_roc(y_true, y_probs, model_name, ax)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "roc_curves_all_models.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    return auc_results


def _plot_single_roc(y_true, y_probs, model_name, ax):
    n_classes = len(CLASS_NAMES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[i]        = auc(fpr[i], tpr[i])

    all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], color=CLASS_COLORS[i], lw=1.5,
                label=f"{CLASS_NAMES[i]}  (AUC = {roc_auc[i]:.2f})")
    ax.plot(all_fpr, mean_tpr, color="black", lw=2.5, linestyle="--",
            label=f"Macro avg  (AUC = {roc_auc['macro']:.2f})")
    ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1, linestyle=":")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(model_name, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return roc_auc


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Usage::

        cam_gen = GradCAM(model, target_layer)
        cam, pred_class = cam_gen.generate(input_tensor)
        overlay = overlay_gradcam(input_tensor, cam)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """Compute the Grad-CAM heat-map.

        Args:
            input_tensor : ``(1, 3, H, W)`` on the same device as the model.
            class_idx    : Target class; uses the predicted class if *None*.

        Returns:
            (cam, predicted_class_idx) where cam is an H×W ndarray in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam).squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_gradcam(
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto the original (normalised) image tensor."""
    img = image_tensor.squeeze().cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    h, w        = img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap     = heatmap[:, :, ::-1] / 255.0   # BGR → RGB, normalise

    overlay = alpha * heatmap + (1 - alpha) * img
    return np.clip(overlay, 0, 1)


def visualize_gradcam_all_models(
    models: dict,
    target_layers: dict,
    image_tensor: torch.Tensor,
    true_label: Optional[int] = None,
):
    """Plot Grad-CAM overlays for multiple models side by side.

    Args:
        models       : ``{"Model Name": model, ...}``
        target_layers: ``{"Model Name": target_layer_module, ...}``
        image_tensor : ``(1, 3, H, W)`` tensor.
        true_label   : Ground-truth class index (used in title only).
    """
    device = next(iter(models.values())).parameters().__next__().device if False else (
        next(iter(models.values())).parameters()
    )
    # Resolve device properly
    first_model = next(iter(models.values()))
    device = next(first_model.parameters()).device
    image_tensor = image_tensor.to(device)

    n = len(models)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

    # Original image
    orig = (image_tensor.squeeze().cpu() * IMAGENET_STD + IMAGENET_MEAN)
    orig = orig.permute(1, 2, 0).numpy()
    orig = np.clip(orig, 0, 1)
    true_str = f"True: {LABEL_NAMES.get(true_label, '?')}" if true_label is not None else ""
    axes[0].imshow(orig)
    axes[0].set_title(f"Original\n{true_str}", fontsize=12)
    axes[0].axis("off")

    for ax, (name, model) in zip(axes[1:], models.items()):
        gradcam = GradCAM(model, target_layers[name])
        cam, pred = gradcam.generate(image_tensor)
        overlay = overlay_gradcam(image_tensor, cam)
        ax.imshow(overlay)
        ax.set_title(f"{name}\nPred: {LABEL_NAMES[pred]}", fontsize=12)
        ax.axis("off")

    plt.suptitle("Grad-CAM Comparison — HAM10000", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
