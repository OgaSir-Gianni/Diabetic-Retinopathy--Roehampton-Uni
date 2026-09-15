"""Generate the dataset figures used in the report (outputs/figures/).

Fig 1 — class distribution bar chart.
Fig 2 — raw vs crop+resize vs Ben Graham, one example per grade.

Usage:  python scripts/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_NAMES, OUTPUT_DIR, RAW_DIR  # noqa: E402
from src.data import make_splits  # noqa: E402
from src.preprocessing import preprocess_image  # noqa: E402

FIG_DIR = OUTPUT_DIR / "figures"


def fig_class_distribution(df: pd.DataFrame) -> None:
    counts = df["diagnosis"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values, color="tab:blue")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 25, f"{v}\n({v / len(df):.0%})", ha="center", fontsize=9)
    ax.set_ylabel("Images")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("APTOS 2019 class distribution (n = 3,662)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_class_distribution.png", dpi=200)
    plt.close(fig)


def fig_preprocessing_examples(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(15, 9.5))
    for k in range(5):
        id_code = df[df["diagnosis"] == k].iloc[0]["id_code"]
        path = RAW_DIR / "train_images" / f"{id_code}.png"
        raw = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        axes[0, k].imshow(raw)
        axes[0, k].set_title(f"{CLASS_NAMES[k]}\nraw", fontsize=10)
        axes[1, k].imshow(preprocess_image(path, apply_ben=False))
        axes[1, k].set_title("crop + resize", fontsize=10)
        axes[2, k].imshow(preprocess_image(path, apply_ben=True))
        axes[2, k].set_title("+ Ben Graham", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_preprocessing.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = make_splits()
    fig_class_distribution(df)
    fig_preprocessing_examples(df)
    print(f"Figures written to {FIG_DIR}")
