# Diabetic Retinopathy Detection with CNNs and Explainability

Deep Learning Applications module — coursework project (APTOS 2019 Blindness Detection).

Five-class diabetic retinopathy (DR) severity grading from retinal fundus photographs,
with a transfer-learning model comparison, a preprocessing ablation, and Grad-CAM
explainability analysis.

## Task

- **Dataset:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) (~3,660 labelled fundus images).
- **Labels:** 0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR.
- **Primary metric:** Quadratic Weighted Kappa (QWK) — the clinical standard for DR grading.
- **Secondary:** per-class F1, confusion matrix, and a binary "referable DR" (grade ≥ 2)
  screening view with sensitivity / specificity / AUC.

## Models compared

| Model | Initialisation | Input size |
|---|---|---|
| ResNet-18 | from scratch (baseline) | 224 |
| EfficientNet-B0 | ImageNet transfer | 224 |
| EfficientNet-B3 | ImageNet transfer | 300 |

All trained with class-weighted cross-entropy to handle the strong class imbalance
(grade 0 is ~half the data).

## Repository layout

```
dr-detection/
├── notebooks/DR_Detection_APTOS.ipynb   # Colab notebook — runs top-to-bottom
├── src/                                 # all library code
│   ├── config.py        # paths, constants, hyperparameter defaults
│   ├── preprocessing.py # circle crop + Ben Graham lighting normalisation
│   ├── data.py          # splits, Dataset, DataLoaders, class weights
│   ├── models.py        # model factory (ResNet-18, EfficientNet-B0/B3)
│   ├── train.py         # training loop (CLI)
│   ├── evaluate.py      # test-set metrics + figures (CLI)
│   ├── gradcam.py       # Grad-CAM implementation + overlay panels (CLI)
│   └── utils.py         # seeding, device selection, plotting
├── scripts/
│   ├── download_data.py   # Kaggle download + verification
│   └── preprocess_data.py # build cached preprocessed image sets
├── report/report_outline.md
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Create a Kaggle API token (Kaggle → Settings → Create New Token) and provide it as
`export KAGGLE_API_TOKEN=KGAT_...` (or save it to `~/.kaggle/access_token`), then:

```bash
python scripts/download_data.py --mirror   # public full-res mirror, no rules acceptance needed
python scripts/preprocess_data.py          # builds data/processed/{ben,plain}/
```

Omit `--mirror` to download the official competition files instead — that requires
having accepted the APTOS 2019 rules on the competition page first. Both sources
yield the same 3,662 labelled images in the same layout (the script verifies the
label distribution against the official counts).

## Reproducing the experiments

```bash
# 1. Baseline: ResNet-18 from scratch
python -m src.train --model resnet18_scratch --variant ben

# 2. Transfer learning
python -m src.train --model efficientnet_b0 --variant ben
python -m src.train --model efficientnet_b3 --variant ben

# 3. Preprocessing ablation (best model, no Ben Graham normalisation)
python -m src.train --model efficientnet_b0 --variant plain

# 4. Test-set evaluation (per run)
python -m src.evaluate --run efficientnet_b0_ben

# 5. Grad-CAM panels (per run)
python -m src.gradcam --run efficientnet_b0_ben
```

Each run writes to `outputs/<run_name>/`: `best.pt`, `history.json`, `metrics.json`,
and figures (learning curves, confusion matrices, Grad-CAM panels).

All experiments are seeded (`--seed`, default 42) and use a stratified 70/15/15
train/val/test split that is identical across runs, so models are compared on the
same held-out test set.

Alternatively, open `notebooks/DR_Detection_APTOS.ipynb` in Google Colab — it clones
this repository and runs the full pipeline end to end on a GPU runtime.
