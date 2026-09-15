"""Held-out test-set evaluation for a completed training run.

Produces, under outputs/<run>/:
    metrics.json           — QWK, accuracy, macro/per-class F1, referable-DR
                             sensitivity/specificity/AUC
    confusion_matrix.png   — 5-class counts + row-normalised
    predictions.csv        — id_code, true grade, predicted grade, probabilities

Usage:
    python -m src.evaluate --run efficientnet_b0_ben
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .config import CLASS_NAMES, DEFAULTS, IMG_SIZE, OUTPUT_DIR, REFERABLE_THRESHOLD
from .data import build_loaders
from .models import build_model
from .train import run_epoch
from .utils import get_device, save_json


def load_run(run: str, device):
    """Rebuild the model from a run's checkpoint; returns (model, ckpt dict)."""
    ckpt = torch.load(OUTPUT_DIR / run / "best.pt", map_location=device,
                      weights_only=True)
    model = build_model(ckpt["model"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def referable_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    """Binary screening view: grades >= REFERABLE_THRESHOLD are 'referable'.

    The referable probability is the summed softmax mass of grades 2-4, so the
    AUC reflects the model's usefulness as a screening triage score."""
    y_bin = (y >= REFERABLE_THRESHOLD).astype(int)
    prob_ref = prob[:, REFERABLE_THRESHOLD:].sum(axis=1)
    pred_bin = (prob_ref >= 0.5).astype(int)
    tp = int(((pred_bin == 1) & (y_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (y_bin == 0)).sum())
    fp = int(((pred_bin == 1) & (y_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (y_bin == 1)).sum())
    return {
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "auc": float(roc_auc_score(y_bin, prob_ref)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def plot_confusion(y: np.ndarray, pred: np.ndarray, out_path, title: str = ""):
    cm = confusion_matrix(y, pred, labels=range(len(CLASS_NAMES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, fmt, sub in [
        (axes[0], cm, "d", "Counts"),
        (axes[1], cm_norm, ".2f", "Row-normalised (recall)"),
    ]:
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted grade")
        ax.set_ylabel("True grade")
        ax.set_title(sub)
        thresh = mat.max() / 2
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(run: str, batch_size: int = DEFAULTS["batch_size"],
             num_workers: int = DEFAULTS["num_workers"]) -> dict:
    device = get_device()
    model, ckpt = load_run(run, device)
    variant = ckpt["variant"]
    loaders = build_loaders(variant, IMG_SIZE[ckpt["model"]], batch_size, num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    _, y, pred, prob = run_epoch(model, loaders["test"], criterion, device)

    metrics = {
        "run": run,
        "test_qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "test_accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_f1": {
            name: float(f)
            for name, f in zip(CLASS_NAMES,
                               f1_score(y, pred, average=None,
                                        labels=range(len(CLASS_NAMES))))
        },
        "referable": referable_metrics(y, prob),
        "n_test": int(len(y)),
    }

    out_dir = OUTPUT_DIR / run
    save_json(metrics, out_dir / "metrics.json")
    plot_confusion(y, pred, out_dir / "confusion_matrix.png", title=run)

    test_df = loaders["split_df"].query("split == 'test'").reset_index(drop=True)
    pred_df = pd.DataFrame({"id_code": test_df["id_code"], "true": y, "pred": pred})
    for i, name in enumerate(CLASS_NAMES):
        pred_df[f"prob_{i}"] = prob[:, i]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print(f"{run}: test QWK {metrics['test_qwk']:.4f} | acc "
          f"{metrics['test_accuracy']:.4f} | macro F1 {metrics['macro_f1']:.4f} | "
          f"referable AUC {metrics['referable']['auc']:.4f}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run folder name, e.g. efficientnet_b0_ben")
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    a = p.parse_args()
    evaluate(a.run, a.batch_size, a.num_workers)
