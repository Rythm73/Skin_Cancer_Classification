 ## Multi-Class Skin Cancer Classification — Deep Benchmarking

---
An end-to-end computer vision system for classifying 7 types of skin lesions using the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dataset. This project benchmarks a custom CNN baseline against fine-tuned ResNet-50 and EfficientNet-B3 architectures, with a specific focus on handling severe class imbalance and ensuring model interpretability through Grad-CAM.

## Abstract
---
Skin cancer is the most common human malignancy. Early and accurate diagnosis is
critical for patient outcomes, yet access to expert dermatologists is unequal
globally. This project trains and compares three deep learning classifiers on the
10,015-image HAM10000 benchmark, tackling the severe **7:1 class imbalance**
through lesion-aware data splitting, minority oversampling, and weighted cross-entropy
loss. Grad-CAM visualisations demonstrate that the best-performing model focuses on
clinically meaningful regions, providing a foundation for explainable medical AI.

---

## Technical Architecture

### Models

| Model | Strategy | Trainable params |
|---|---|---|
| **Custom CNN** | 3-block conv baseline trained from scratch | ~170 K |
| **ResNet-50** | ImageNet pre-trained; last residual block (`layer4`) unfrozen | ~25 M / 23 M frozen |
| **EfficientNet-B3** | ImageNet pre-trained; last 3 feature blocks unfrozen | ~12 M / 10 M frozen |

### Data Strategy

- **Lesion-aware split** (70 / 15 / 15) — images from the same lesion *never* span
  train and test, preventing data leakage.
- **Minority oversampling** — under-represented classes are upsampled to 500 images
  each in the training set.
- **Weighted cross-entropy** — `sklearn.utils.class_weight.compute_class_weight`
  generates per-class weights to further penalise majority-class errors.
- **Augmentation** — random horizontal/vertical flip, 30° rotation, colour jitter,
  and affine translate applied at train time only.

### Interpretability

Gradient-weighted Class Activation Mapping (**Grad-CAM**) is used to visualise which
image regions each model attends to. Target layers:

| Model | Target layer |
|---|---|
| EfficientNet-B3 | `features[-1][0]` — last Conv2d before avgpool |
| ResNet-50 | `layer4[-1].conv3` — last conv in final Bottleneck |
| Custom CNN | `features[8]` — Conv2d(64→128) in Block 3 |

---

## Key Results

| Metric | Custom CNN | ResNet-50 | EfficientNet-B3 |
|---|---|---|---|
| Accuracy | — | — | **~76%** |
| F1 Weighted | — | — | — |
| F1 Macro | — | — | **~0.67** |
| ROC-AUC (macro) | — | — | — |

=================================================================
  FINAL MODEL COMPARISON
=================================================================
  Metric                      CNN     ResNet   EfficientNet
  ------------------------------------------------------------
  Accuracy                 0.4459     0.7606         0.7876
  F1 Weighted              0.5118     0.7793         0.8020
  F1 Macro                 0.2506     0.6332         0.6774
  Precision                0.7245     0.8157         0.8285
  Recall                   0.4459     0.7606         0.7876
  ROC-AUC                  0.8479     0.9455         0.9552
=================================================================

> Fill in your own numbers after training. EfficientNet-B3 consistently outperforms
> both baselines, particularly on minority classes (akiec, df, vasc).

The confusion matrix reveals the hardest pair: **mel vs. nv** — both are pigmented
lesions that share visual features at the pixel level.

---

## Project Structure

```
skin-cancer-classification/
├── data/               # Download instructions (raw images excluded from repo)
│   └── README.md
├── notebooks/          # Exploratory data analysis
│   └── exploration.ipynb
├── src/                # Core source code
│   ├── __init__.py
│   ├── dataset.py      # SkinLesionDataset, transforms, split & oversample helpers
│   ├── models.py       # CustomCNN, ResNet-50, EfficientNet-B3 definitions
│   ├── train.py        # Training loop + CLI entry point
│   └── utils.py        # Metrics, plotting, Grad-CAM
├── models/             # Saved checkpoints (.pt) — excluded from Git
├── predict.py          # Single-image inference script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/skin-cancer-classification.git
cd skin-cancer-classification
pip install -r requirements.txt
```

### 2. Download the dataset

See [`data/README.md`](data/README.md) for full instructions.

### 3. Train a model

```bash
# Train EfficientNet-B3 (recommended)
python src/train.py --data_dir data/ --model efficientnet --epochs 35

# Train ResNet-50
python src/train.py --data_dir data/ --model resnet50 --epochs 35 --lr 1e-4

# Train the Custom CNN baseline
python src/train.py --data_dir data/ --model cnn --epochs 50 --lr 1e-3
```

All checkpoints are saved to `models/best_<model_name>.pt`.

### 4. Predict on a single image

```bash
# Basic prediction
python predict.py --image path/to/lesion.jpg \
                  --model efficientnet \
                  --checkpoint models/best_efficientnet.pt

# With Grad-CAM overlay
python predict.py --image path/to/lesion.jpg \
                  --model efficientnet \
                  --checkpoint models/best_efficientnet.pt \
                  --gradcam
```

---

## CLI Reference — `train.py`

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Path to HAM10000 directory |
| `--model` | `efficientnet` | `cnn` / `resnet50` / `efficientnet` |
| `--epochs` | `35` | Number of training epochs |
| `--lr` | `1e-4` | Initial learning rate (Adam) |
| `--batch_size` | `32` | Mini-batch size |
| `--oversample_target` | `500` | Target count per class after oversampling |
| `--checkpoint_dir` | `models/` | Where to save `.pt` checkpoints |
| `--no_class_weights` | `False` | Disable weighted cross-entropy |
| `--img_size` | `300` | Input image size (px) |

---

## Experiment Tracking *(optional)*

Plug in [Weights & Biases](https://wandb.ai) or [MLflow](https://mlflow.org) to log
loss curves, hyperparameters, and confusion matrices:

```bash
pip install wandb
wandb login
# Then add --use_wandb flag (implementation left as exercise)
```

---

## Disclaimer

This project is for **research and educational purposes only**. It is not a medical
device and should not be used for clinical diagnosis. Always consult a qualified
dermatologist.

---

## License

MIT
