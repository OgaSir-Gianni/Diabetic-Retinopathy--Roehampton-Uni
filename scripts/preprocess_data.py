"""Build the cached preprocessed image sets (both ablation variants).

Preprocessing (circle crop + resize + optional Ben Graham) is deterministic,
so it is done once here rather than on every epoch:

    data/processed/ben/     — crop + resize + Ben Graham normalisation
    data/processed/plain/   — crop + resize only (ablation control)

Usage:  python scripts/preprocess_data.py
"""

import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CACHE_SIZE, RAW_DIR  # noqa: E402
from src.data import processed_dir  # noqa: E402
from src.preprocessing import preprocess_image  # noqa: E402


def build_variant(ids: list[str], variant: str) -> None:
    apply_ben = variant == "ben"
    out_dir = processed_dir(variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (out_dir / f"{i}.png").exists()]
    print(f"[{variant}] {len(todo)} images to process "
          f"({len(ids) - len(todo)} already cached)")
    for id_code in tqdm(todo, desc=variant):
        img = preprocess_image(
            RAW_DIR / "train_images" / f"{id_code}.png",
            size=CACHE_SIZE,
            apply_ben=apply_ben,
        )
        cv2.imwrite(str(out_dir / f"{id_code}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main() -> None:
    df = pd.read_csv(RAW_DIR / "train.csv")
    ids = df["id_code"].tolist()
    for variant in ["ben", "plain"]:
        build_variant(ids, variant)
    print("Preprocessing cache complete.")


if __name__ == "__main__":
    main()
