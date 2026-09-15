"""Download the APTOS 2019 dataset from Kaggle into data/raw/.

Two sources, same resulting layout (data/raw/train.csv + data/raw/train_images/):

- Official competition data (default). Requires having ACCEPTED the APTOS 2019
  competition rules on the Kaggle website (competition page -> Late Submission /
  Join Competition), otherwise the API returns 403 Forbidden.
- ``--mirror``: the public full-resolution mirror ``mariaherrerot/aptos2019``
  (no rules acceptance needed). It re-splits the same 3,662 labelled images
  into train/valid/test folders; this script merges them back into the
  official single-folder layout. Our own stratified split (src/data.py) is
  applied afterwards either way, so the two sources are interchangeable.

Auth: a Kaggle API token (kaggle.com -> Settings -> Create New Token), provided
as the KAGGLE_API_TOKEN env var (the "KGAT_..." string, or a path to a file
containing it), a token file at ~/.kaggle/access_token, or legacy
~/.kaggle/kaggle.json. Requires kaggle CLI >= 2.2 for KGAT tokens.

Usage:
    python scripts/download_data.py            # official competition data
    python scripts/download_data.py --mirror   # public mirror
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, KAGGLE_COMPETITION, RAW_DIR  # noqa: E402

MIRROR_DATASET = "mariaherrerot/aptos2019"
# Official APTOS 2019 training-label distribution, used to verify integrity.
OFFICIAL_COUNTS = {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295}


def ensure_kaggle_token() -> None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.exists():
        os.environ["KAGGLE_API_TOKEN"] = str(token_file)
        return
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return  # legacy username+key auth
    sys.exit(
        "No Kaggle token found. Create one at kaggle.com -> Settings -> "
        "Create New Token, then either export KAGGLE_API_TOKEN=<token> or "
        "save it to ~/.kaggle/access_token."
    )


def kaggle_cli() -> str:
    """Prefer the kaggle CLI installed next to this interpreter (venv-safe)."""
    candidate = Path(sys.executable).with_name("kaggle")
    return str(candidate) if candidate.exists() else "kaggle"


def download_official() -> None:
    zip_path = RAW_DIR / f"{KAGGLE_COMPETITION}.zip"
    if not zip_path.exists():
        print("Downloading official competition data (~9.5 GB)...")
        subprocess.run(
            [kaggle_cli(), "competitions", "download", "-c", KAGGLE_COMPETITION,
             "-p", str(RAW_DIR)],
            check=True,
        )
    # Extract only what we use: the competition's test images are
    # unlabelled (no public ground truth), so skip them to save ~13 GB.
    print("Unzipping train files...")
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist()
                   if n == "train.csv" or n.startswith("train_images/")]
        z.extractall(RAW_DIR, members=members)
    zip_path.unlink()  # reclaim disk space


def download_mirror() -> None:
    import pandas as pd

    mirror_dir = DATA_DIR / "mirror"
    if not any(mirror_dir.glob("*.csv")):
        mirror_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading mirror {MIRROR_DATASET} (~8.6 GB)...")
        subprocess.run(
            [kaggle_cli(), "datasets", "download", "-d", MIRROR_DATASET,
             "-p", str(mirror_dir)],
            check=True,
        )
        zip_path = next(mirror_dir.glob("*.zip"))
        print("Unzipping...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(mirror_dir)
        zip_path.unlink()

    # Merge the mirror's train/valid/test re-split back into the official
    # single-folder layout; our own stratified split is applied later.
    print("Consolidating mirror layout into data/raw/ ...")
    img_dir = RAW_DIR / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(f)[["id_code", "diagnosis"]]
              for f in sorted(mirror_dir.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("id_code")
    for png in mirror_dir.rglob("*.png"):
        dest = img_dir / png.name
        if not dest.exists():
            shutil.move(str(png), dest)
    df.to_csv(RAW_DIR / "train.csv", index=False)
    shutil.rmtree(mirror_dir)  # only empty folders/CSV copies remain


def verify() -> None:
    import pandas as pd

    df = pd.read_csv(RAW_DIR / "train.csv")
    img_dir = RAW_DIR / "train_images"
    missing = [i for i in df["id_code"] if not (img_dir / f"{i}.png").exists()]
    counts = df["diagnosis"].value_counts().sort_index().to_dict()
    print(f"train.csv rows: {len(df)} | missing images: {len(missing)}")
    print(f"Label distribution: {counts}")
    if missing:
        sys.exit(f"ERROR: {len(missing)} images referenced but not found.")
    if counts != OFFICIAL_COUNTS:
        print(f"WARNING: distribution differs from official {OFFICIAL_COUNTS}")
    print("Dataset OK.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", action="store_true",
                        help="use the public mirror dataset (no competition "
                             "rules acceptance needed)")
    args = parser.parse_args()

    ensure_kaggle_token()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not (RAW_DIR / "train.csv").exists():
        download_mirror() if args.mirror else download_official()
    verify()


if __name__ == "__main__":
    main()
