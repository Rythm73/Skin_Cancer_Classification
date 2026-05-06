"""
predict.py — Single-image inference script.

Usage
-----
Run prediction on one image::

    python predict.py --image /path/to/lesion.jpg --model efficientnet \\
                      --checkpoint models/best_efficientnet.pt

With Grad-CAM overlay::

    python predict.py --image /path/to/lesion.jpg --model efficientnet \\
                      --checkpoint models/best_efficientnet.pt --gradcam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import LABEL_NAMES, LABEL_MAP, get_val_test_transforms
from models import get_model, build_resnet50, build_efficientnet_b3, CustomCNN
from utils import GradCAM, overlay_gradcam


CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}


def load_trained_model(model_name: str, checkpoint_path: str, device: torch.device):
    """Rebuild architecture and load saved weights from a training checkpoint."""
    model = get_model(model_name)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def get_gradcam_target(model_name: str, model):
    """Return the target layer for Grad-CAM based on architecture."""
    targets = {
        "efficientnet": model.features[-1][0],
        "resnet50":     model.layer4[-1].conv3,
        "cnn":          model.features[8],
    }
    if model_name not in targets:
        raise ValueError(f"No Grad-CAM target defined for '{model_name}'.")
    return targets[model_name]


def predict(image_path: str, model, device: torch.device, img_size: int = 300):
    """Run inference on a single image.

    Returns:
        (predicted_label_str, confidence_float, probabilities_tensor)
    """
    transform = get_val_test_transforms(img_size)
    image     = Image.open(image_path).convert("RGB")
    tensor    = transform(image).unsqueeze(0).to(device)   # (1, 3, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()

    pred_idx    = probs.argmax().item()
    pred_label  = LABEL_NAMES[pred_idx]
    confidence  = probs[pred_idx].item()
    return pred_label, confidence, probs, tensor


def print_results(pred_label: str, confidence: float, probs):
    print("\n" + "=" * 50)
    print("  PREDICTION RESULT")
    print("=" * 50)
    print(f"  Predicted class : {pred_label.upper()}")
    print(f"  Description     : {CLASS_DESCRIPTIONS[pred_label]}")
    print(f"  Confidence      : {confidence * 100:.1f}%")
    print("\n  Class probabilities:")
    for idx, name in LABEL_NAMES.items():
        bar = "█" * int(probs[idx].item() * 30)
        print(f"    {name:<6}  {probs[idx].item() * 100:5.1f}%  {bar}")
    print("=" * 50)
    print("\n  ⚠  This tool is for research purposes only.")
    print("     Always consult a qualified dermatologist.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image skin-lesion classification."
    )
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to the input skin-lesion image.")
    parser.add_argument("--model",      type=str, default="efficientnet",
                        choices=["cnn", "resnet50", "efficientnet"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the saved .pt checkpoint file.")
    parser.add_argument("--img_size",   type=int, default=300)
    parser.add_argument("--gradcam",    action="store_true",
                        help="Display a Grad-CAM overlay alongside the prediction.")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model '{args.model}' from {args.checkpoint} …")
    model = load_trained_model(args.model, args.checkpoint, device)

    pred_label, confidence, probs, tensor = predict(
        args.image, model, device, args.img_size
    )
    print_results(pred_label, confidence, probs)

    if args.gradcam:
        target_layer = get_gradcam_target(args.model, model)
        gradcam      = GradCAM(model, target_layer)
        cam, _       = gradcam.generate(tensor)
        overlay      = overlay_gradcam(tensor, cam)

        orig_image = Image.open(args.image).convert("RGB")
        fig, axes  = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(orig_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM — Pred: {pred_label} ({confidence * 100:.1f}%)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
