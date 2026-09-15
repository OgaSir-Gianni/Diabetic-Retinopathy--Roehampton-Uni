"""Fundus image preprocessing.

Two steps, both standard in the Kaggle DR competition literature:

1. Circle crop — fundus photographs are a bright circular retina on a black
   background whose margins vary by camera; cropping to the retina's bounding
   box removes uninformative black borders before resizing.
2. Ben Graham normalisation — subtract a heavily Gaussian-blurred copy of the
   image from itself (``4*img - 4*blur + 128``). The blur estimates the local
   illumination field, so subtracting it flattens lighting differences across
   cameras/clinics and visually enhances vessels and lesions.

The training pipeline reads from a cache built by ``scripts/preprocess_data.py``
in two variants ("ben" and "plain") so the preprocessing ablation trains on
identical images apart from the normalisation step.
"""

from pathlib import Path

import cv2
import numpy as np

from .config import BEN_SIGMA, CACHE_SIZE


def crop_to_retina(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop away near-black borders, keeping the bounding box of the retina.

    ``tol`` is the grey-level threshold below which a pixel counts as
    background. Falls back to the full image if the mask comes out empty
    (e.g. an unusually dark photograph).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return img
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img[r0 : r1 + 1, c0 : c1 + 1]


def ben_graham(img: np.ndarray, sigma: float = BEN_SIGMA) -> np.ndarray:
    """Ben Graham Gaussian-blur subtraction (local illumination removal)."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def preprocess_image(
    path: str | Path, size: int = CACHE_SIZE, apply_ben: bool = True
) -> np.ndarray:
    """Full pipeline for one image: load -> crop -> resize -> (Ben Graham).

    Returns an RGB uint8 array of shape (size, size, 3).
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = crop_to_retina(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if apply_ben:
        img = ben_graham(img)
    return img
