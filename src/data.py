"""Dataset splits, PyTorch Dataset/DataLoaders, and class weights.

The split is stratified by diagnosis and written once to ``data/splits.csv``;
every training run reads the same file, so all models are trained and compared
on identical train/val/test partitions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    PROCESSED_DIR,
    RAW_DIR,
    SEED,
    SPLIT_CSV,
    TEST_FRACTION,
    VAL_FRACTION,
)


def make_splits(seed: int = SEED, force: bool = False) -> pd.DataFrame:
    """Create (or load) the frozen stratified train/val/test split."""
    if SPLIT_CSV.exists() and not force:
        return pd.read_csv(SPLIT_CSV)

    df = pd.read_csv(RAW_DIR / "train.csv")  # columns: id_code, diagnosis
    trainval, test = train_test_split(
        df, test_size=TEST_FRACTION, stratify=df["diagnosis"], random_state=seed
    )
    # Val fraction is relative to the full dataset.
    rel_val = VAL_FRACTION / (1 - TEST_FRACTION)
    train, val = train_test_split(
        trainval, test_size=rel_val, stratify=trainval["diagnosis"], random_state=seed
    )
    df = df.copy()
    df["split"] = "train"
    df.loc[df["id_code"].isin(val["id_code"]), "split"] = "val"
    df.loc[df["id_code"].isin(test["id_code"]), "split"] = "test"

    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLIT_CSV, index=False)
    return df


class APTOSDataset(Dataset):
    """Reads cached preprocessed PNGs (see scripts/preprocess_data.py)."""

    def __init__(self, df: pd.DataFrame, variant: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = processed_dir(variant)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / f"{row.id_code}.png").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row.diagnosis)


def processed_dir(variant: str) -> Path:
    """Cache folder for a preprocessing variant ('ben' or 'plain')."""
    return PROCESSED_DIR / variant


def build_transforms(img_size: int) -> tuple:
    """(train, eval) transforms. Augmentation is conservative and clinically
    plausible: fundus orientation is arbitrary, so flips/rotations are safe."""
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1.

    Grade 0 is ~half of APTOS; unweighted cross-entropy lets the model buy
    accuracy by over-predicting it, which QWK punishes."""
    counts = train_df["diagnosis"].value_counts().reindex(range(NUM_CLASSES)).fillna(0)
    weights = len(train_df) / (NUM_CLASSES * counts.clip(lower=1))
    weights = weights / weights.mean()
    return torch.tensor(weights.values, dtype=torch.float32)


def build_loaders(
    variant: str,
    img_size: int,
    batch_size: int,
    num_workers: int = 2,
    seed: int = SEED,
) -> dict:
    """DataLoaders for train/val/test plus the class-weight tensor."""
    df = make_splits(seed=seed)
    train_tf, eval_tf = build_transforms(img_size)

    parts = {}
    for split, tf, shuffle in [
        ("train", train_tf, True),
        ("val", eval_tf, False),
        ("test", eval_tf, False),
    ]:
        ds = APTOSDataset(df[df["split"] == split], variant, transform=tf)
        parts[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # no-op warning on MPS/CPU otherwise
            drop_last=(split == "train"),
        )

    parts["class_weights"] = class_weights(df[df["split"] == "train"])
    parts["split_df"] = df
    return parts
