"""Grad-CAM (Selvaraju et al., 2017) for fundus images.

Answers the report's explainability question: when the model grades an image,
is it looking at actual DR lesions (microaneurysms, haemorrhages, exudates)
or at artefacts? Panels deliberately include failure cases and
"right answer, wrong evidence" cases — those drive the Discussion section.

Run AFTER src.evaluate (it reuses outputs/<run>/predictions.csv):
    python -m src.gradcam --run efficientnet_b0_ben

Produces, under outputs/<run>/gradcam/:
    correct_grade<k>.png   — highest-confidence correct example per grade
    errors.png             — highest-confidence misclassifications
"""

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import CLASS_NAMES, IMG_SIZE, OUTPUT_DIR
from .data import build_transforms, processed_dir
from .evaluate import load_run
from .models import gradcam_target_layer
from .utils import get_device


class GradCAM:
    """Minimal Grad-CAM: class-gradient-weighted average of the target layer's
    activation maps, ReLU'd and upsampled to the input size."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """Returns (heatmap in [0,1] at input resolution, predicted class)."""
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP of grads
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return ((1 - alpha) * img_rgb + alpha * heat).astype(np.uint8)


def _panel(rows: pd.DataFrame, cam_engine, eval_tf, img_dir, img_size,
           device, out_path, title):
    """One figure: original | Grad-CAM overlay for each selected image."""
    n = len(rows)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n), squeeze=False)
    for i, (_, r) in enumerate(rows.iterrows()):
        pil = Image.open(img_dir / f"{r.id_code}.png").convert("RGB")
        x = eval_tf(pil).unsqueeze(0).to(device)
        cam, _ = cam_engine(x, class_idx=int(r.pred))
        disp = np.array(pil.resize((img_size, img_size)))

        axes[i, 0].imshow(disp)
        axes[i, 0].set_title(f"{r.id_code}\ntrue: {CLASS_NAMES[int(r.true)]}",
                             fontsize=9)
        axes[i, 1].imshow(overlay(disp, cam))
        conf = r[f"prob_{int(r.pred)}"]
        axes[i, 1].set_title(
            f"pred: {CLASS_NAMES[int(r.pred)]} (p={conf:.2f})", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_panels(run: str, per_class: int = 2, n_errors: int = 6):
    device = get_device()
    model, ckpt = load_run(run, device)
    img_size = IMG_SIZE[ckpt["model"]]
    _, eval_tf = build_transforms(img_size)
    img_dir = processed_dir(ckpt["variant"])

    pred_csv = OUTPUT_DIR / run / "predictions.csv"
    if not pred_csv.exists():
        raise FileNotFoundError(
            f"{pred_csv} not found — run `python -m src.evaluate --run {run}` first.")
    df = pd.read_csv(pred_csv)
    df["conf"] = df.apply(lambda r: r[f"prob_{int(r.pred)}"], axis=1)

    cam_engine = GradCAM(model, gradcam_target_layer(model, ckpt["model"]))
    out = OUTPUT_DIR / run / "gradcam"

    # Highest-confidence correct predictions, per grade.
    for k, name in enumerate(CLASS_NAMES):
        rows = (df[(df.true == k) & (df.pred == k)]
                .nlargest(per_class, "conf"))
        _panel(rows, cam_engine, eval_tf, img_dir, img_size, device,
               out / f"correct_grade{k}.png",
               f"{run} — correct: {name}")

    # Highest-confidence mistakes: where the model is confidently wrong.
    errors = df[df.true != df.pred].nlargest(n_errors, "conf")
    _panel(errors, cam_engine, eval_tf, img_dir, img_size, device,
           out / "errors.png", f"{run} — confident misclassifications")
    print(f"Grad-CAM panels written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--n-errors", type=int, default=6)
    a = p.parse_args()
    generate_panels(a.run, a.per_class, a.n_errors)
