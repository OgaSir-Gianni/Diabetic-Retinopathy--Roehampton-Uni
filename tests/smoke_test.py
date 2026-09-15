"""End-to-end smoke test on a synthetic mini-dataset — no Kaggle data needed.

Builds 60 fake fundus-like images (a noisy bright circle on black), then runs
the ENTIRE pipeline exactly as Colab will: preprocessing cache -> stratified
split -> 2 training epochs (ResNet-18 scratch) -> test evaluation -> Grad-CAM
panels. Catches wiring bugs before spending real GPU time.

Usage:  python tests/smoke_test.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Redirect all data/output paths into a throwaway dir BEFORE importing src.
_tmp = tempfile.mkdtemp(prefix="dr_smoke_")
os.environ["DR_DATA_DIR"] = str(Path(_tmp) / "data")
os.environ["DR_OUTPUT_DIR"] = str(Path(_tmp) / "outputs")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.preprocessing import ben_graham, crop_to_retina, preprocess_image  # noqa: E402

N_PER_CLASS = 12
RUN = "resnet18_scratch_ben"


def make_fake_dataset() -> None:
    """Synthetic fundus-like images: bright noisy circle on black background."""
    rng = np.random.default_rng(0)
    img_dir = config.RAW_DIR / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(config.NUM_CLASSES):
        for i in range(N_PER_CLASS):
            id_code = f"fake_{k}_{i}"
            img = np.zeros((280, 320, 3), np.uint8)
            cv2.circle(img, (160, 140), 120,
                       tuple(int(c) for c in rng.integers(90, 200, 3)), -1)
            noise = rng.integers(0, 60, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
            cv2.imwrite(str(img_dir / f"{id_code}.png"), img)
            rows.append({"id_code": id_code, "diagnosis": k})
    pd.DataFrame(rows).to_csv(config.RAW_DIR / "train.csv", index=False)
    print(f"[1/6] fake dataset: {len(rows)} images at {config.RAW_DIR}")


def test_preprocessing() -> None:
    sample = next((config.RAW_DIR / "train_images").glob("*.png"))
    img = cv2.cvtColor(cv2.imread(str(sample)), cv2.COLOR_BGR2RGB)
    cropped = crop_to_retina(img)
    assert cropped.shape[0] <= img.shape[0] and cropped.shape[1] <= img.shape[1]
    out = preprocess_image(sample, size=128, apply_ben=True)
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8
    assert not np.array_equal(out, preprocess_image(sample, size=128, apply_ben=False))
    _ = ben_graham(img)
    print("[2/6] preprocessing OK")


def build_caches() -> None:
    df = pd.read_csv(config.RAW_DIR / "train.csv")
    for variant, ben in [("ben", True), ("plain", False)]:
        out_dir = config.PROCESSED_DIR / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        for id_code in df["id_code"]:
            img = preprocess_image(config.RAW_DIR / "train_images" / f"{id_code}.png",
                                   size=config.CACHE_SIZE, apply_ben=ben)
            cv2.imwrite(str(out_dir / f"{id_code}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("[3/6] preprocessing caches built (ben + plain)")


def test_models() -> None:
    import torch

    from src.models import build_model, gradcam_target_layer

    for name in ["resnet18_scratch"]:  # pretrained builds are exercised on Colab
        m = build_model(name)
        y = m(torch.randn(2, 3, 128, 128))
        assert y.shape == (2, config.NUM_CLASSES)
        gradcam_target_layer(m, name)
    print("[4/6] model factory OK")


def test_train_eval_gradcam() -> None:
    from src.evaluate import evaluate
    from src.gradcam import generate_panels
    from src.train import parse_args, train

    args = parse_args([
        "--model", "resnet18_scratch", "--variant", "ben",
        "--epochs", "2", "--batch-size", "8", "--num-workers", "0",
    ])
    history = train(args)
    assert len(history["train_loss"]) == 2
    assert (config.OUTPUT_DIR / RUN / "best.pt").exists()
    assert (config.OUTPUT_DIR / RUN / "curves.png").exists()
    print("[5/6] training loop OK")

    metrics = evaluate(RUN, batch_size=8, num_workers=0)
    assert -1.0 <= metrics["test_qwk"] <= 1.0
    assert set(metrics["per_class_f1"]) == set(config.CLASS_NAMES)
    assert (config.OUTPUT_DIR / RUN / "confusion_matrix.png").exists()
    assert (config.OUTPUT_DIR / RUN / "predictions.csv").exists()

    generate_panels(RUN, per_class=1, n_errors=2)
    panels = list((config.OUTPUT_DIR / RUN / "gradcam").glob("*.png"))
    assert panels, "no Grad-CAM panels produced"
    print(f"[6/6] evaluate + Grad-CAM OK ({len(panels)} panels)")


if __name__ == "__main__":
    make_fake_dataset()
    test_preprocessing()
    build_caches()
    test_models()
    test_train_eval_gradcam()
    print(f"\nALL SMOKE TESTS PASSED (workspace: {_tmp})")
