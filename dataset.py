"""
dataset.py — Custom Dataset class, label map, and transforms for HAM10000.
"""

import os
import glob
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ── Label mapping ──────────────────────────────────────────────────────────────
LABEL_MAP: dict[str, int] = {
    "akiec": 0,
    "bcc":   1,
    "bkl":   2,
    "df":    3,
    "mel":   4,
    "nv":    5,
    "vasc":  6,
}
LABEL_NAMES: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = list(LABEL_MAP.keys())

# ── ImageNet stats (used for all three models) ─────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 300) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_test_transforms(img_size: int = 300) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class SkinLesionDataset(Dataset):
    """PyTorch Dataset for the HAM10000 skin-lesion benchmark.

    Args:
        dataframe (pd.DataFrame): Must contain columns ``path`` (absolute image
            path) and ``dx`` (diagnosis label string).
        transform: Optional torchvision transform pipeline.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        super().__init__()
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        img_path  = self.dataframe.loc[idx, "path"]
        image     = Image.open(img_path).convert("RGB")
        label_str = self.dataframe.loc[idx, "dx"]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(LABEL_MAP[label_str], dtype=torch.long)
        return image, label


# ── Data loading helpers ───────────────────────────────────────────────────────

def load_metadata(data_dir: str) -> pd.DataFrame:
    """Read the HAM10000 CSV and resolve each image_id to an absolute path.

    Expects the following layout inside *data_dir*::

        data_dir/
        ├── HAM10000_metadata.csv
        ├── HAM10000_images_part_1/   *.jpg
        └── HAM10000_images_part_2/   *.jpg

    Returns:
        pd.DataFrame with columns: lesion_id, image_id, dx, path
    """
    csv_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    df = pd.read_csv(csv_path)

    all_images = glob.glob(os.path.join(data_dir, "HAM10000_images_part_*", "*.jpg"))
    image_path_map = {
        os.path.splitext(os.path.basename(p))[0]: p for p in all_images
    }

    df["path"] = df["image_id"].map(image_path_map)
    df = df[["lesion_id", "image_id", "dx", "path"]]

    missing = df["path"].isna().sum()
    if missing:
        print(f"[WARNING] {missing} rows have no matching image file.")
    return df


def split_by_lesion(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lesion-aware train / val / test split (prevents data leakage).

    Splits on unique *lesion_id* so that images from the same lesion
    never appear in more than one split.
    """
    lesions = df["lesion_id"].unique()
    temp_frac = val_size + test_size

    train_lesions, temp_lesions = train_test_split(
        lesions, test_size=temp_frac, random_state=random_state
    )
    relative_test = test_size / temp_frac
    val_lesions, test_lesions = train_test_split(
        temp_lesions, test_size=relative_test, random_state=random_state
    )

    train_df = df[df["lesion_id"].isin(train_lesions)].copy()
    val_df   = df[df["lesion_id"].isin(val_lesions)].copy()
    test_df  = df[df["lesion_id"].isin(test_lesions)].copy()
    return train_df, val_df, test_df


def oversample_minority_classes(
    train_df: pd.DataFrame,
    target_count: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Upsample under-represented classes to *target_count* per class."""
    oversampled_dfs = []
    for cls in train_df["dx"].unique():
        cls_df  = train_df[train_df["dx"] == cls]
        current = len(cls_df)
        if current < target_count:
            extra = cls_df.sample(
                n=target_count - current, replace=True, random_state=random_state
            )
            cls_df = pd.concat([cls_df, extra])
        oversampled_dfs.append(cls_df)

    return (
        pd.concat(oversampled_dfs)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
