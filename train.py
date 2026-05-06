"""
train.py — Main training script for HAM10000 skin-lesion classification.

Usage examples
--------------
Train EfficientNet-B3 for 35 epochs::

    python src/train.py --data_dir /path/to/HAM10000 --model efficientnet --epochs 35

Train ResNet-50 with a custom learning rate::

    python src/train.py --data_dir /path/to/HAM10000 --model resnet50 --lr 5e-5

Train the custom CNN baseline::

    python src/train.py --data_dir /path/to/HAM10000 --model cnn --epochs 50

All checkpoints are saved under ``models/`` by default.
"""

from __future__ import annotations

import argparse
import copy
import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# ── local imports ──────────────────────────────────────────────────────────────
from dataset import (
    load_metadata,
    split_by_lesion,
    oversample_minority_classes,
    SkinLesionDataset,
    get_train_transforms,
    get_val_test_transforms,
    CLASS_NAMES,
)
from models import get_model
from utils import (
    evaluate_model,
    plot_training_curves,
    plot_model_comparison,
)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    checkpoint_dir: str = "models",
) -> tuple[nn.Module, pd.DataFrame]:
    """Train *model* and save the best checkpoint by validation loss.

    Args:
        model          : PyTorch model (moved to *device* internally).
        model_name     : Used for checkpoint filename and log messages.
        train_loader   : DataLoader for the training split.
        val_loader     : DataLoader for the validation split.
        epochs         : Number of training epochs.
        lr             : Initial learning rate (Adam).
        device         : ``torch.device``.
        class_weights  : Optional 1-D tensor of per-class weights on *device*.
        checkpoint_dir : Directory where ``best_<model_name>.pt`` is saved.

    Returns:
        (best_model, history_dataframe)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(device))
        if class_weights is not None
        else nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "val_f1":     [],
    }
    best_val_loss  = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    ckpt_path      = os.path.join(checkpoint_dir, f"best_{model_name}.pt")

    print(f"\nTraining: {model_name}")
    print("=" * 50)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_true, train_pred = [], []

        for inputs, labels in tqdm(train_loader, desc="  Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            train_true.extend(labels.cpu().numpy())
            train_pred.extend(preds.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc      = accuracy_score(train_true, train_pred)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_true, val_pred = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="  Val  "):
                inputs, labels = inputs.to(device), labels.to(device)
                logits  = model(inputs)
                loss    = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1)
                val_true.extend(labels.cpu().numpy())
                val_pred.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc      = accuracy_score(val_true, val_pred)
        val_f1       = f1_score(val_true, val_pred, average="weighted", zero_division=0)

        scheduler.step()

        # ── Checkpoint ─────────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             avg_val_loss,
                    "val_acc":              val_acc,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best model → {ckpt_path}")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"  Train → Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}\n"
            f"  Val   → Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
        )

    model.load_state_dict(best_model_wts)
    print(f"\nDone — {model_name} | Best Val Loss: {best_val_loss:.4f}")
    return model, pd.DataFrame(history)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a skin-lesion classifier on HAM10000."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the HAM10000 dataset directory.",
    )
    parser.add_argument(
        "--model", type=str, default="efficientnet",
        choices=["cnn", "resnet50", "efficientnet"],
        help="Model architecture to train (default: efficientnet).",
    )
    parser.add_argument(
        "--epochs", type=int, default=35,
        help="Number of training epochs (default: 35).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--oversample_target", type=int, default=500,
        help="Target per-class count for oversampling (default: 500).",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="models",
        help="Directory to save model checkpoints (default: models/).",
    )
    parser.add_argument(
        "--no_class_weights", action="store_true",
        help="Disable weighted cross-entropy loss.",
    )
    parser.add_argument(
        "--img_size", type=int, default=300,
        help="Image resize dimension (default: 300).",
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load & split data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading metadata …")
    df = load_metadata(args.data_dir)
    train_df, val_df, test_df = split_by_lesion(df)
    train_df_balanced = oversample_minority_classes(train_df, args.oversample_target)

    print(f"  Train (balanced): {len(train_df_balanced)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print("\n[2/5] Building datasets …")
    train_transforms = get_train_transforms(args.img_size)
    val_transforms   = get_val_test_transforms(args.img_size)

    train_dataset = SkinLesionDataset(train_df_balanced, transform=train_transforms)
    val_dataset   = SkinLesionDataset(val_df,            transform=val_transforms)
    test_dataset  = SkinLesionDataset(test_df,           transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights_tensor = None
    if not args.no_class_weights:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(sorted(train_df["dx"].unique())),
            y=train_df["dx"].values,
        )
        class_weights_tensor = torch.tensor(weights, dtype=torch.float)
        print("\n[3/5] Class weights:")
        for cls, w in zip(sorted(train_df["dx"].unique()), weights):
            print(f"  {cls:<8} → {w:.4f}")
    else:
        print("\n[3/5] Skipping class weights (--no_class_weights).")

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Building model: {args.model} …")
    model = get_model(args.model)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training …")
    trained_model, history = train_model(
        model          = model,
        model_name     = args.model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        epochs         = args.epochs,
        lr             = args.lr,
        device         = device,
        class_weights  = class_weights_tensor,
        checkpoint_dir = args.checkpoint_dir,
    )

    plot_training_curves(history, args.model, save_dir=args.checkpoint_dir)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n── Test-set evaluation ──")
    evaluate_model(trained_model, test_loader, args.model, device,
                   save_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
