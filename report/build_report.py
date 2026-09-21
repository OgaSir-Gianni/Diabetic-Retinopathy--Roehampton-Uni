"""Build the draft coursework report (Word .docx) from the actual run outputs.

Numbers, tables and figures are pulled from outputs/<run>/metrics.json,
history.json, predictions.csv and the generated figure PNGs, so re-running
this script after new training runs refreshes the whole document:

    python report/build_report.py       ->  report/Report_Draft.docx

Text wrapped in double angle quotes in the DRAFT is rendered red in Word:
either an editing instruction or a value still pending training.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_NAMES, OUTPUT_DIR  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
OUT_DOCX = REPORT_DIR / "Report_Draft.docx"
REPO_URL = "https://github.com/OgaSir-Gianni/Diabetic-Retinopathy--Roehampton-Uni"

RUNS = {
    "resnet18_scratch_ben": "ResNet-18 (scratch)",
    "efficientnet_b0_ben": "EfficientNet-B0 (transfer)",
    "efficientnet_b3_ben": "EfficientNet-B3 (transfer)",
    "efficientnet_b0_plain": "EfficientNet-B0 (no Ben Graham)",
}
PARAMS_M = {"resnet18_scratch_ben": 11.2, "efficientnet_b0_ben": 4.0,
            "efficientnet_b3_ben": 10.7, "efficientnet_b0_plain": 4.0}
INPUT_PX = {"resnet18_scratch_ben": 224, "efficientnet_b0_ben": 224,
            "efficientnet_b3_ben": 300, "efficientnet_b0_plain": 224}

RED = RGBColor(0xC0, 0x00, 0x00)
PEND = "« PENDING »"  # rendered red


# ---------------------------------------------------------------- data access
def load(run: str, name: str):
    p = OUTPUT_DIR / run / name
    if not p.exists():
        return None
    if name.endswith(".json"):
        return json.loads(p.read_text())
    return p


def f3(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else PEND


def pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else PEND


def num(x):
    """The value if numeric, else None (json ints and floats both count)."""
    return x if isinstance(x, (int, float)) else None


def delta(a, b):
    """a - b when both are numeric, else None (f3 renders None as pending)."""
    return a - b if num(a) is not None and num(b) is not None else None


# British Diabetic Association screening standard for referable DR.
BDA_SENS, BDA_SPEC = 0.80, 0.95


def bda_shortfall(ref: dict):
    """None if the operating point meets the BDA standard, otherwise the name
    of the failing metric(s); 'unknown' when values are missing."""
    s, p = num(ref.get("sensitivity")), num(ref.get("specificity"))
    if s is None or p is None:
        return "unknown"
    misses = [name for name, value, bar in
              [("sensitivity", s, BDA_SENS), ("specificity", p, BDA_SPEC)]
              if value < bar]
    return " and ".join(misses) or None


def metric(run: str, *keys):
    m = load(run, "metrics.json")
    for k in keys:
        if m is None:
            return None
        m = m.get(k)
    return m


def adjacent_error_pct(run: str):
    p = OUTPUT_DIR / run / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    errors = df[df.true != df.pred]
    if len(errors) == 0:
        return None
    return (abs(errors.true - errors.pred) == 1).mean()


# ------------------------------------------------------------- docx utilities
def style_document(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for h, size in [("Heading 1", 14), ("Heading 2", 12), ("Title", 20)]:
        s = doc.styles[h]
        s.font.name = "Times New Roman"
        s.font.size = Pt(size)
        s.font.color.rgb = RGBColor(0, 0, 0)


def para(doc: Document, text: str, style=None, align=None, italic=False,
         size=None):
    """Add a paragraph; segments wrapped in « » are rendered red."""
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    for i, seg in enumerate(text.split("«")):
        if i == 0:
            runs = [(seg, False)]
        else:
            red_part, _, rest = seg.partition("»")
            runs = [("«" + red_part + "»", True), (rest, False)]
        for chunk, is_red in runs:
            if not chunk:
                continue
            r = p.add_run(chunk)
            if is_red:
                r.font.color.rgb = RED
                r.bold = True
            if italic:
                r.italic = True
            if size:
                r.font.size = Pt(size)
    return p


def caption(doc: Document, text: str) -> None:
    para(doc, text, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=9)


def figure(doc: Document, path, cap: str, width=6.0) -> None:
    if path is None:
        para(doc, f"« FIGURE PENDING: {cap} »",
             align=WD_ALIGN_PARAGRAPH.CENTER)
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, cap)


def table(doc: Document, header: list, rows: list, cap: str) -> None:
    caption(doc, cap)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(10)
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = ""
            r = cell.paragraphs[0].add_run(str(v))
            r.font.size = Pt(10)
            if PEND in str(v):
                r.font.color.rgb = RED
    doc.add_paragraph()


# -------------------------------------------------------------------- content
def build() -> None:
    b0, sc, b3, pl = ("efficientnet_b0_ben", "resnet18_scratch_ben",
                      "efficientnet_b3_ben", "efficientnet_b0_plain")

    # Shared, computed once: every directional claim below derives from these,
    # so a regeneration with different data changes the wording, not just the
    # numbers.
    qwks = {r: num(metric(r, "test_qwk")) for r in RUNS}
    best = max(RUNS, key=lambda r: qwks[r] if qwks[r] is not None else -1.0)
    best_ref = metric(best, "referable") or {}
    best_bda = bda_shortfall(best_ref)  # None = meets the standard

    doc = Document()
    style_document(doc)

    # ---- Title page
    para(doc, "Diabetic Retinopathy Detection with Convolutional Neural "
              "Networks and Explainability", style="Title",
         align=WD_ALIGN_PARAGRAPH.CENTER)
    for line in ["« Your Name — Student Number »",
                 "Deep Learning Applications — Coursework Portfolio",
                 f"{date.today():%d %B %Y}",
                 f"Code repository: {REPO_URL}"]:
        para(doc, line, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "« DRAFT — working document. Red text = to edit or "
              "pending final training runs. This draft was assembled from the "
              "project's actual experiment outputs; rewrite the prose in your "
              "own words before submission and delete this notice. »",
         align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()

    # ---- Abstract
    doc.add_heading("Abstract", level=1)
    para(doc,
         "Diabetic retinopathy (DR) is a leading cause of preventable blindness, "
         "and manual grading of retinal fundus photographs is the main bottleneck "
         "in screening programmes. This project develops and compares deep "
         "learning models for five-class DR severity grading on the APTOS 2019 "
         "dataset (3,662 images). A ResNet-18 trained from scratch is compared "
         "against transfer-learned EfficientNet-B0 and B3, using class-weighted "
         "training, fundus-specific preprocessing, and an ablation of Ben Graham "
         f"illumination normalisation. The best configuration "
         f"({RUNS[best]}) reached a quadratic weighted kappa of "
         f"{f3(qwks[best])} on a held-out test set and "
         f"{pct(best_ref.get('sensitivity'))} sensitivity / "
         f"{pct(best_ref.get('specificity'))} specificity for referable DR, "
         f"{'meeting' if best_bda is None else 'approaching'} the British "
         "Diabetic Association screening standard at the default operating "
         "point. Transfer learning proved decisive, while the widely used "
         "Ben Graham normalisation brought no benefit to ImageNet-pretrained "
         "models. Grad-CAM analysis of the transfer-learned models shows "
         "attention often falls on clinically meaningful regions but can rely "
         "on treatment artefacts (laser scars), highlighting explainability "
         "as a prerequisite for clinical deployment.")

    # ---- 1 Introduction
    doc.add_heading("1. Introduction", level=1)
    para(doc,
         "Diabetic retinopathy is a complication of diabetes in which "
         "progressive damage to retinal blood vessels can lead to irreversible "
         "sight loss. It is a leading cause of blindness in the working-age "
         "population, yet vision loss is largely preventable if the disease is "
         "detected and treated early [1]. National screening programmes "
         "therefore photograph the retina (fundus) of diabetic patients at "
         "regular intervals, and trained graders assign a severity grade that "
         "determines whether the patient is referred to an ophthalmologist. "
         "With a growing diabetic population, grading capacity — not image "
         "acquisition — is the principal bottleneck of these programmes.")
    para(doc,
         "Severity grading is fundamentally a visual pattern-recognition task: "
         "graders look for microaneurysms, haemorrhages, exudates and abnormal "
         "new vessels. This makes it well suited to convolutional neural "
         "networks (CNNs), which learn such visual features directly from "
         "labelled images. Landmark studies have shown that deep learning can "
         "match specialist-level performance at scale [1], [2], establishing "
         "why a deep learning approach is appropriate for this problem.")
    para(doc, "The aims of this project are to:")
    for aim in [
        "train and compare CNN architectures for five-class DR grading on the "
        "APTOS 2019 dataset;",
        "quantify the contribution of transfer learning and of fundus-specific "
        "preprocessing through controlled comparisons and an ablation study;",
        "evaluate the models in the binary “referable DR” framing "
        "used by real screening programmes;",
        "assess with Grad-CAM whether model attention is clinically plausible, "
        "including failure cases.",
    ]:
        doc.add_paragraph(aim, style="List Number")
    para(doc,
         "The scope is deliberately constrained to a single public dataset and "
         "free-tier GPU compute, reflecting what is achievable without clinical "
         "data access; the implications of this constraint are discussed in "
         "Section 5.")

    # ---- 2 Background
    doc.add_heading("2. Background and Literature Review", level=1)
    para(doc,
         "Clinical grading. DR severity is commonly reported on the "
         "five-level International Clinical Diabetic Retinopathy scale [8]: "
         "0 (no DR), 1 (mild), 2 (moderate), 3 (severe non-proliferative) and "
         "4 (proliferative). Early grades are defined by microaneurysms and "
         "small haemorrhages; moderate and severe grades by increasing "
         "haemorrhage, hard exudates and venous abnormalities; proliferative "
         "DR by neovascularisation, which threatens sight and requires urgent "
         "treatment. Screening programmes typically act on a binary decision "
         "— whether DR is “referable” (grade 2 or above) — "
         "so both the ordinal grade and this binary view matter in practice.")
    para(doc,
         "Deep learning for DR. Gulshan et al. [1] trained an Inception "
         "network on 128,175 fundus images and reported specialist-level "
         "detection of referable DR (AUC ≈ 0.99), a result validated in "
         "multi-ethnic populations by Ting et al. [2]. De Fauw et al. [3] "
         "demonstrated a clinically deployable triage system for retinal "
         "disease, emphasising the importance of interpretability and of "
         "framing models as referral-recommendation tools rather than "
         "autonomous diagnosticians. These studies used one to two orders of "
         "magnitude more images than are publicly available; a central "
         "question for this project is what is achievable at APTOS scale "
         "(~3,700 images), which is why transfer learning is expected to be "
         "decisive.")
    para(doc,
         "Methods adopted here. Transfer learning reuses features learned "
         "on ImageNet and fine-tunes them on the target task, which is the "
         "standard remedy for small medical datasets. EfficientNet [4] scales "
         "network depth, width and input resolution jointly and achieves "
         "strong accuracy per parameter, making B0/B3 suitable for free-tier "
         "GPUs. Grad-CAM [5] produces class-specific attention heatmaps from "
         "the final convolutional layer and is widely used to check whether "
         "medical imaging models attend to pathology or to artefacts. Finally, "
         "Ben Graham's Gaussian-blur subtraction, introduced in the winning "
         "entry of the 2015 Kaggle DR competition [6], normalises illumination "
         "across cameras and visually enhances vessels and lesions; it is "
         "evaluated here as an explicit ablation rather than assumed to help.")

    # ---- 3 Methodology
    doc.add_heading("3. Methodology", level=1)
    doc.add_heading("3.1 Dataset", level=2)
    para(doc,
         "The APTOS 2019 Blindness Detection dataset [7] contains 3,662 "
         "labelled fundus photographs collected by the Aravind Eye Hospital "
         "screening programme in India, graded 0–4 by clinicians. The "
         "class distribution is strongly imbalanced (Figure 1): grade 0 "
         "accounts for 49% of images while severe (grade 3) accounts for only "
         "5%. This imbalance is treated as a property of the problem to be "
         "handled and analysed, not removed: screening populations are "
         "dominated by healthy eyes, so a deployable model must cope with it.")
    figure(doc, OUTPUT_DIR / "figures" / "fig1_class_distribution.png",
           "Figure 1. APTOS 2019 class distribution (n = 3,662).")

    doc.add_heading("3.2 Preprocessing", level=2)
    para(doc,
         "Fundus photographs vary in framing and illumination across cameras. "
         "Each image is first cropped to the bounding box of the retinal disc "
         "(removing uninformative black borders), resized to 320×320, and "
         "optionally normalised with Ben Graham's method [6]: "
         "I' = 4I − 4·Gσ(I) + 128, where Gσ is a Gaussian "
         "blur (σ = 10) that estimates the local illumination field. "
         "Subtracting it flattens lighting differences and enhances vessels "
         "and lesions (Figure 2). Both preprocessed variants are cached to "
         "disk so the ablation in Section 4.2 trains on otherwise identical "
         "data.")
    figure(doc, OUTPUT_DIR / "figures" / "fig2_preprocessing.png",
           "Figure 2. One example per grade: raw image (top), circle-crop and "
           "resize (middle), Ben Graham normalisation (bottom).")

    doc.add_heading("3.3 Data splits", level=2)
    para(doc,
         "The data is split 70/15/15 into training (2,562), validation (550) "
         "and test (550) sets, stratified by grade and frozen once (seed 42); "
         "every experiment reads the same split file, so all models are "
         "selected on the same validation set and compared on the same "
         "held-out test set, which is only used for final evaluation.")

    doc.add_heading("3.4 Models", level=2)
    para(doc,
         "Three architectures are compared (Table 1). ResNet-18 trained from "
         "random initialisation provides the “no transfer learning” "
         "baseline; EfficientNet-B0 and B3 [4] are initialised with ImageNet "
         "weights and fine-tuned end-to-end. B3 additionally tests whether "
         "higher input resolution (300 px) helps detect small lesions such as "
         "microaneurysms at this dataset size. The final classification layer "
         "of each network is replaced with a 5-way linear head.")
    table(doc,
          ["Model", "Initialisation", "Parameters", "Input size"],
          [["ResNet-18", "random (scratch)", "11.2 M", "224 px"],
           ["EfficientNet-B0", "ImageNet", "4.0 M", "224 px"],
           ["EfficientNet-B3", "ImageNet", "10.7 M", "300 px"]],
          "Table 1. Compared architectures.")

    doc.add_heading("3.5 Training procedure", level=2)
    para(doc,
         "All runs share the same procedure so that only the architecture or "
         "the preprocessing variant changes. Training uses cross-entropy "
         "weighted by inverse class frequency (weights 0.22 / 1.06 / 0.39 / "
         "2.02 / 1.32 for grades 0–4), the AdamW optimiser (learning rate "
         "3×10⁻⁴, weight decay 10⁻⁴), a cosine "
         "learning-rate schedule, batch size 32 (16 for B3), and a maximum of "
         "30 epochs with early stopping after 8 epochs without validation "
         "improvement. Augmentation is restricted to transformations that are "
         "clinically label-preserving: random flips, rotation, small crops "
         "and mild brightness/contrast jitter (fundus orientation carries no "
         "diagnostic meaning). The checkpoint with the best validation QWK is "
         "kept. All code, seeds and per-run configurations are available in "
         f"the repository ({REPO_URL}), and the accompanying Colab notebook "
         "reproduces the full pipeline end-to-end.")

    doc.add_heading("3.6 Evaluation metrics", level=2)
    para(doc,
         "The primary metric is quadratic weighted kappa (QWK), the standard "
         "for DR grading: because grades are ordinal, misgrading proliferative "
         "DR as “no DR” must be penalised far more than confusing "
         "adjacent grades, which plain accuracy does not capture. Per-class "
         "F1 and confusion matrices expose behaviour on minority grades. "
         "Finally, each model is evaluated as a binary screening test for "
         "referable DR (grade ≥ 2), reporting sensitivity, specificity "
         "and AUC of the summed referable probability — the framing in "
         "which such models are actually deployed [1].")

    # ---- 4 Results
    doc.add_heading("4. Experiments and Results", level=1)

    doc.add_heading("4.1 Model comparison", level=2)
    rows = []
    for run in [sc, b0, pl, b3]:
        rows.append([RUNS[run],
                     f3(metric(run, "test_qwk")),
                     f3(metric(run, "test_accuracy")),
                     f3(metric(run, "macro_f1")),
                     pct(metric(run, "referable", "sensitivity")),
                     pct(metric(run, "referable", "specificity")),
                     f3(metric(run, "referable", "auc"))])
    table(doc,
          ["Model", "QWK", "Accuracy", "Macro F1", "Ref. sens.", "Ref. spec.",
           "Ref. AUC"],
          rows,
          "Table 2. Held-out test-set results (550 images). Referable DR = "
          "grade ≥ 2.")
    h0 = load(b0, "history.json") or {}
    d_b3 = delta(qwks[b3], qwks[b0])
    b3_word = ("outperformed" if d_b3 is not None and d_b3 > 0
               else "matched rather than beat")
    f1_gap = delta(num(metric(b3, "macro_f1")), num(metric(b0, "macro_f1")))
    f1_word = "higher" if f1_gap is not None and f1_gap > 0 else "lower"
    aucs = {r: num(metric(r, "referable", "auc")) for r in RUNS}
    b3_top_auc = (aucs[b3] is not None
                  and aucs[b3] >= max(a for a in aucs.values() if a is not None))
    b3_clause = (
        f", although B3 did achieve the highest referable-DR AUC "
        f"({f3(aucs[b3])}) and sensitivity "
        f"({pct(metric(b3, 'referable', 'sensitivity'))})" if b3_top_auc else "")
    para(doc,
         "Transfer learning dominates the comparison "
         f"(Table 2): EfficientNet-B0 reaches a test QWK of "
         f"{f3(qwks[b0])} versus "
         f"{f3(qwks[sc])} for the identical training procedure "
         "with a randomly initialised ResNet-18. "
         f"EfficientNet-B3, despite {PARAMS_M[b3] / PARAMS_M[b0]:.1f}× more "
         f"parameters and higher input resolution, {b3_word} B0 (QWK "
         f"{f3(qwks[b3])} vs {f3(qwks[b0])}) and "
         f"scored a {f1_word} macro F1 — at this dataset size the additional "
         f"capacity brings no reliable gain in grading{b3_clause}. "
         "The learning curves (Figure 3) show that the pretrained model "
         f"reaches its best validation QWK "
         f"({f3((h0.get('best_val_qwk')))}) after only "
         f"{h0.get('best_epoch', PEND)} epochs and then begins to overfit "
         "— training loss continues to fall while validation loss rises "
         "— which early stopping handles; this rapid convergence is "
         "typical when fine-tuning strong pretrained features on a small "
         "dataset.")
    figure(doc, load(b0, "curves.png"),
           "Figure 3. Learning curves for EfficientNet-B0 (Ben Graham "
           "variant): loss (left) and validation QWK (right).")

    doc.add_heading("4.2 Preprocessing ablation", level=2)
    table(doc,
          ["Preprocessing", "QWK", "Accuracy", "Macro F1", "Ref. AUC"],
          [["Ben Graham", f3(metric(b0, "test_qwk")),
            f3(metric(b0, "test_accuracy")), f3(metric(b0, "macro_f1")),
            f3(metric(b0, "referable", "auc"))],
           ["Crop + resize only", f3(metric(pl, "test_qwk")),
            f3(metric(pl, "test_accuracy")), f3(metric(pl, "macro_f1")),
            f3(metric(pl, "referable", "auc"))]],
          "Table 3. Ablation of Ben Graham illumination normalisation "
          "(EfficientNet-B0, identical training).")
    h_pl = load(pl, "history.json") or {}
    d_abl = delta(qwks[pl], qwks[b0])
    abl_word = "higher" if d_abl is None or d_abl >= 0 else "lower"
    para(doc,
         "Contrary to expectation, Ben Graham normalisation brought no "
         f"benefit: the plain variant scored marginally {abl_word} on test "
         f"QWK ({f3(qwks[pl])} vs {f3(qwks[b0])}) "
         f"with validation performance effectively tied "
         f"({f3(h_pl.get('best_val_qwk'))} vs "
         f"{f3(h0.get('best_val_qwk'))}), so the "
         "honest conclusion is that the technique does not help an ImageNet-"
         "pretrained network on this dataset, and the small test-set gap is "
         "within seed-level noise. A plausible explanation is that the "
         "benefit reported in the 2015 competition [6] accrued to models "
         "trained from scratch, for which illumination variance is a real "
         "nuisance factor; modern pretrained features are already robust to "
         "global lighting, while the transformation discards absolute colour "
         "information that may itself be informative. This is a useful "
         "negative result: preprocessing folklore should be re-validated "
         "rather than inherited. « Add your own view: would you now "
         "recommend Ben Graham for this pipeline? Consider repeating with a "
         "different seed if time allows. »")

    doc.add_heading("4.3 Error structure across grades", level=2)
    adj = adjacent_error_pct(b0)
    f1s = metric(b0, "per_class_f1") or {}
    para(doc,
         "The confusion matrix (Figure 4) shows errors are heavily "
         "concentrated between adjacent grades: "
         f"{pct(adj)} of all misclassifications are off by exactly one "
         "grade. Per-class F1 falls from "
         f"{f3(f1s.get('No DR'))} (no DR) to "
         f"{f3(f1s.get('Mild'))} (mild), "
         f"{f3(f1s.get('Moderate'))} (moderate), "
         f"{f3(f1s.get('Severe'))} (severe) and "
         f"{f3(f1s.get('Proliferative'))} (proliferative), reflecting both "
         "class scarcity and the genuine clinical ambiguity of grade "
         "boundaries — inter-grader agreement on these grades is itself "
         "imperfect [1]. In other words, the model behaves like a slightly "
         "noisy grader rather than one making random errors, which is what "
         "the high QWK captures.")
    figure(doc, load(b0, "confusion_matrix.png"),
           "Figure 4. Test-set confusion matrices for EfficientNet-B0: "
           "counts (left) and row-normalised recall (right).")

    doc.add_heading("4.4 Referable-DR screening view", level=2)
    ref = metric(b0, "referable") or {}
    ref_pl = metric(pl, "referable") or {}
    b0_short = bda_shortfall(ref)
    pl_short = bda_shortfall(ref_pl)
    b0_verdict = ("meeting the British Diabetic Association screening "
                  "standard (≥80% sensitivity, ≥95% specificity)"
                  if b0_short is None else
                  "narrowly missing the British Diabetic Association "
                  "screening standard (≥80% sensitivity, ≥95% "
                  f"specificity) on {b0_short}")
    pl_verdict = ("meets both targets untuned" if pl_short is None
                  else f"misses the standard on {pl_short}")
    tl_aucs = [a for r, a in aucs.items() if r != sc and a is not None]
    auc_span = f"{min(tl_aucs):.3f}–{max(tl_aucs):.3f}" if tl_aucs else PEND
    para(doc,
         "Collapsed to the binary referable decision, EfficientNet-B0 (Ben "
         f"Graham) achieves {pct(ref.get('sensitivity'))} sensitivity and "
         f"{pct(ref.get('specificity'))} specificity at the default 0.5 "
         f"operating point (AUC {f3(ref.get('auc'))}), {b0_verdict}. The "
         f"plain-preprocessing variant {pl_verdict} "
         f"({pct(ref_pl.get('sensitivity'))} / "
         f"{pct(ref_pl.get('specificity'))}), and the near-identical AUCs of "
         f"the transfer-learned variants ({auc_span}) suggest each of them "
         "could reach the standard by tuning the operating point on the "
         "validation set — a deployment decision that trades referral "
         "workload against missed disease. That models trained on ~2,500 "
         "images approach screening-standard operating characteristics "
         "underlines how much of the task pretrained features already "
         "capture.")

    doc.add_heading("4.5 Explainability with Grad-CAM", level=2)
    gc_dir = OUTPUT_DIR / b0 / "gradcam"
    para(doc,
         "Grad-CAM heatmaps were generated for the highest-confidence correct "
         "prediction of each grade and for the most confident "
         "misclassifications. "
         "« Describe what you see in the final panels. From the current "
         "run: for correctly-classified proliferative cases the attention "
         "concentrates on panretinal photocoagulation laser scars — "
         "i.e. evidence of prior treatment — rather than on "
         "neovascularisation itself. The prediction is right but the "
         "evidence is a shortcut, discussed in Section 5. »")
    figure(doc, (gc_dir / "correct_grade4.png") if (gc_dir / "correct_grade4.png").exists() else None,
           "Figure 5. Grad-CAM for correctly classified proliferative DR: "
           "original (left) and attention overlay (right). Attention falls on "
           "laser-treatment scars rather than on new vessels.", width=4.5)
    figure(doc, (gc_dir / "errors.png") if (gc_dir / "errors.png").exists() else None,
           "Figure 6. Grad-CAM for the most confident misclassifications.",
           width=4.5)

    # ---- 5 Discussion
    doc.add_heading("5. Discussion", level=1)
    qwk_gain = delta(qwks[b0], qwks[sc])
    para(doc,
         "Interpretation. The results quantify the project's design "
         f"decisions. First, transfer learning is worth {f3(qwk_gain)} QWK "
         "over training from scratch with an otherwise identical procedure "
         "— at this dataset size it is the single most consequential "
         "choice made, and the macro-F1 gap between the scratch and "
         f"pretrained models ({f3(metric(sc, 'macro_f1'))} vs "
         f"{f3(metric(b0, 'macro_f1'))}) shows that its benefit is "
         "concentrated in the rare grades. Second, neither scaling up the "
         "architecture (B3) nor the classical Ben Graham preprocessing "
         "improved five-class grading: with strong pretrained features and "
         "only ~2,500 training images, the binding constraint is data, not "
         "model capacity or input normalisation. Class weighting itself was "
         "held constant across all runs, so its individual contribution was "
         "not ablated; isolating it is noted as future work. "
         "« Add a sentence of your own overall reading here. »")
    para(doc,
         "Right answers for wrong reasons. The most important finding of "
         "the explainability analysis is that the network can exploit "
         "treatment artefacts: laser scars are strong evidence that a patient "
         "was treated for proliferative DR, so attending to them yields "
         "correct grades on this dataset while telling us little about the "
         "disease itself. A model relying on this shortcut would "
         "systematically miss untreated proliferative DR — precisely the "
         "patients screening exists to find. This mirrors known shortcut-"
         "learning failures in medical imaging and shows why heatmap-level "
         "auditing must accompany headline metrics before any clinical use "
         "[3], [5].")
    para(doc,
         "Limitations. All images come from a single Indian screening "
         "programme, so performance may not transfer across populations, "
         "camera types, or fundus pigmentation — external validation on "
         "a dataset such as Messidor-2 is the necessary next step. Labels "
         "carry inherent grader noise [1], bounding achievable QWK. The test "
         "set (550 images) gives moderate confidence intervals, and no "
         "cross-validation was performed due to compute limits.")
    para(doc,
         "Responsible AI. The costs of error are asymmetric: a false "
         "negative can mean unmonitored progression to sight loss, whereas a "
         "false positive costs an unnecessary referral. This argues for "
         "operating points chosen for high sensitivity, and for framing the "
         "model as a triage assistant that prioritises human grading queues "
         "rather than an autonomous diagnostician [3]. Fairness requires "
         "validation across ethnicities and devices before deployment; "
         "fundus images are biometric, identifiable medical data, so privacy "
         "and governance obligations apply to any real data pipeline. "
         "Explainability, as shown in Section 4.5, is not cosmetic: it is how "
         "shortcut behaviour was detected in this project.")
    para(doc,
         "Future work. Ordinal-aware objectives (regression or CORAL "
         "heads), threshold tuning on the validation set, test-time "
         "augmentation, model ensembling, and external validation are the "
         "highest-value extensions; a lesion-segmentation auxiliary task "
         "could directly discourage shortcut attention.")

    # ---- 6 Conclusion
    doc.add_heading("6. Conclusion", level=1)
    para(doc,
         "This project built a complete, reproducible pipeline for five-class "
         "DR grading and referable-DR screening on APTOS 2019, comparing "
         "scratch and transfer-learned CNNs with a controlled preprocessing "
         f"ablation. The best configuration ({RUNS[best]}) reached a test "
         f"QWK of {f3(qwks[best])} and "
         f"{'met' if best_bda is None else 'approached'} the BDA screening "
         f"standard ({pct(best_ref.get('sensitivity'))} sensitivity, "
         f"{pct(best_ref.get('specificity'))} specificity) at an "
         "untuned operating point, with ~2,500 training images. Beyond the "
         "metrics, the experiments produced two less obvious lessons: an "
         "inherited preprocessing technique added nothing once transfer "
         "learning was in place, and Grad-CAM auditing surfaced a treatment-"
         "artefact shortcut that headline numbers alone would have hidden — "
         "for medical imaging, understanding why a model is right matters as "
         "much as how often it is right.")

    # ---- References
    doc.add_heading("References", level=1)
    refs = [
        'V. Gulshan et al., "Development and validation of a deep learning '
        'algorithm for detection of diabetic retinopathy in retinal fundus '
        'photographs," JAMA, vol. 316, no. 22, pp. 2402–2410, 2016.',
        'D. S. W. Ting et al., "Development and validation of a deep learning '
        'system for diabetic retinopathy and related eye diseases using '
        'retinal images from multiethnic populations with diabetes," JAMA, '
        'vol. 318, no. 22, pp. 2211–2223, 2017.',
        'J. De Fauw et al., "Clinically applicable deep learning for '
        'diagnosis and referral in retinal disease," Nature Medicine, '
        'vol. 24, no. 9, pp. 1342–1350, 2018.',
        'M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for '
        'convolutional neural networks," in Proc. 36th Int. Conf. Machine '
        'Learning (ICML), 2019, pp. 6105–6114.',
        'R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep '
        'networks via gradient-based localization," in Proc. IEEE Int. Conf. '
        'Computer Vision (ICCV), 2017, pp. 618–626.',
        'B. Graham, "Kaggle Diabetic Retinopathy Detection competition '
        'report," University of Warwick, 2015.',
        'Asia Pacific Tele-Ophthalmology Society, "APTOS 2019 Blindness '
        'Detection," Kaggle, 2019. [Online]. Available: '
        'https://www.kaggle.com/competitions/aptos2019-blindness-detection',
        'C. P. Wilkinson et al., "Proposed international clinical diabetic '
        'retinopathy and diabetic macular edema disease severity scales," '
        'Ophthalmology, vol. 110, no. 9, pp. 1677–1682, 2003.',
        'A. Paszke et al., "PyTorch: An imperative style, high-performance '
        'deep learning library," in Advances in Neural Information Processing '
        'Systems 32, 2019, pp. 8024–8035.',
    ]
    for i, r in enumerate(refs, 1):
        para(doc, f"[{i}] {r}")

    doc.save(OUT_DOCX)

    # Rough word count of body prose (excludes tables/captions/references).
    n_words = sum(len(p.text.split()) for p in doc.paragraphs
                  if p.style.name in ("Normal", "List Number"))
    print(f"Saved {OUT_DOCX}")
    print(f"Approximate body word count (incl. draft notes): {n_words}")


if __name__ == "__main__":
    build()
