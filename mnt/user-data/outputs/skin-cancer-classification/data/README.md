# Dataset Setup

This project uses the **HAM10000** (*Human Against Machine with 10,000 training images*)
dataset from the ISIC 2018 challenge.

> Raw images are **not** committed to this repository. Follow the steps below to
> download and arrange them locally.

---

## Expected directory layout

```
data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   ├── ISIC_0024306.jpg
│   └── ...
└── HAM10000_images_part_2/
    ├── ISIC_0029306.jpg
    └── ...
```

---

## Download instructions

### Option A — Kaggle CLI (recommended)

```bash
pip install kaggle
# Place your kaggle.json token at ~/.kaggle/kaggle.json
kaggle datasets download kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/
```

### Option B — ISIC Archive (official source)

1. Create a free account at <https://www.isic-archive.com>.
2. Navigate to **ISIC 2018 — Task 3** and download:
   - `ISIC2018_Task3_Training_Input.zip`
   - `ISIC2018_Task3_Training_GroundTruth.zip`
3. Unzip both archives into `data/`.

---

## Class distribution

| Code  | Full name                                      | # images |
|-------|------------------------------------------------|----------|
| nv    | Melanocytic Nevi                               | 6,705    |
| mel   | Melanoma                                       | 1,113    |
| bkl   | Benign Keratosis-like Lesions                  | 1,099    |
| bcc   | Basal Cell Carcinoma                           | 514      |
| akiec | Actinic Keratoses / Intraepithelial Carcinoma  | 327      |
| vasc  | Vascular Lesions                               | 142      |
| df    | Dermatofibroma                                 | 115      |

The dataset is **severely imbalanced** — `nv` alone accounts for ~67% of samples.
The training pipeline handles this via lesion-aware splitting, minority
oversampling, and weighted cross-entropy loss.
