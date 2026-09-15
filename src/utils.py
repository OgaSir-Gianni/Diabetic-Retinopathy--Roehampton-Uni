"""Shared helpers: reproducibility, device selection, plotting."""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; notebooks re-enable inline backends
import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def plot_history(history: dict, out_path: str | Path, title: str = "") -> None:
    """Learning curves: loss and validation QWK per epoch, side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Cross-entropy loss")
    ax1.legend()

    ax2.plot(epochs, history["val_qwk"], color="tab:green")
    best = int(np.argmax(history["val_qwk"]))
    ax2.scatter([best + 1], [history["val_qwk"][best]], color="tab:red", zorder=5,
                label=f"best = {history['val_qwk'][best]:.3f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Quadratic Weighted Kappa")
    ax2.set_title("Validation QWK")
    ax2.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
