"""Training loop with class-weighted cross-entropy, cosine LR schedule, and
early stopping on validation QWK.

Usage:
    python -m src.train --model efficientnet_b0 --variant ben
    python -m src.train --model efficientnet_b0 --variant plain   # ablation
    python -m src.train --model resnet18_scratch --variant ben

Each run writes to outputs/<model>_<variant>/:
    best.pt        — weights at the best validation-QWK epoch
    history.json   — per-epoch losses and metrics
    curves.png     — learning curves
    config.json    — the exact hyperparameters used
"""

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from .config import DEFAULTS, IMG_SIZE, OUTPUT_DIR, SEED
from .data import build_loaders
from .models import build_model
from .utils import get_device, plot_history, save_json, seed_everything


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evals.

    Returns (mean loss, labels, predictions, softmax probabilities)."""
    training = optimizer is not None
    model.train(training)
    losses, all_y, all_pred, all_prob = [], [], [], []

    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False, desc="train" if training else "eval"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            all_y.append(y.cpu().numpy())
            all_prob.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            all_pred.append(logits.argmax(1).detach().cpu().numpy())

    return (
        float(np.mean(losses)),
        np.concatenate(all_y),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def train(args) -> dict:
    seed_everything(args.seed)
    device = get_device()
    run_name = f"{args.model}_{args.variant}"
    out_dir = OUTPUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_name} | device: {device}")

    img_size = IMG_SIZE[args.model]
    loaders = build_loaders(
        args.variant, img_size, args.batch_size, args.num_workers, seed=args.seed
    )
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(weight=loaders["class_weights"].to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "val_qwk": [], "lr": []}
    best_qwk, best_state, best_epoch = -1.0, None, -1
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, *_ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, y, pred, _ = run_epoch(model, loaders["val"], criterion, device)
        val_qwk = cohen_kappa_score(y, pred, weights="quadratic")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_qwk"].append(float(val_qwk))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        print(
            f"epoch {epoch:02d}/{args.epochs} | train {train_loss:.4f} "
            f"| val {val_loss:.4f} | val QWK {val_qwk:.4f}"
        )

        if val_qwk > best_qwk:
            best_qwk, best_epoch = val_qwk, epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"model": args.model, "variant": args.variant,
                 "epoch": epoch, "val_qwk": float(val_qwk),
                 "state_dict": best_state},
                out_dir / "best.pt",
            )
        elif epoch - best_epoch >= args.early_stop_patience:
            print(f"Early stopping: no val QWK improvement for "
                  f"{args.early_stop_patience} epochs.")
            break

    elapsed = time.time() - start
    history["best_val_qwk"] = best_qwk
    history["best_epoch"] = best_epoch
    history["train_minutes"] = round(elapsed / 60, 1)
    save_json(history, out_dir / "history.json")
    save_json(vars(args), out_dir / "config.json")
    plot_history(history, out_dir / "curves.png", title=run_name)
    print(f"Done in {elapsed/60:.1f} min. Best val QWK {best_qwk:.4f} "
          f"(epoch {best_epoch}). Saved to {out_dir}")
    return history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   choices=["resnet18_scratch", "resnet18",
                            "efficientnet_b0", "efficientnet_b3"])
    p.add_argument("--variant", default="ben", choices=["ben", "plain"],
                   help="preprocessing variant (ablation switch)")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--early-stop-patience", type=int,
                   default=DEFAULTS["early_stop_patience"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
