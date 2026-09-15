# Report outline — Diabetic Retinopathy Detection with CNNs and Explainability

Target: **3,000 words**, IEEE citations, Word/PDF via Turnitin, due **24 Sep 2026**.
Word budgets below sum to ~3,000 excluding title/references/captions.

Rubric mapping: Background 20% · Methodology 25% · **Results & Analysis 30%** ·
Report quality 15% · Originality 10%. The Results/Discussion sections get the
largest budget deliberately.

---

## Title page (not counted)
Project title, name/student number, module name, submission date.
**Include the GitHub repo / Colab link here and repeat it in the Methodology.**

## Abstract (~100 words, optional but include it)
One sentence each: problem, dataset, approach (3 CNNs + ablation + Grad-CAM),
headline result (best test QWK + referable sensitivity/specificity), key finding
from the explainability analysis.

## 1. Introduction (~350 words) — rubric: Understanding 20%
- DR is a leading cause of preventable blindness; screening works but grading
  capacity is the bottleneck (cite IDF/WHO figure for scale).
- Why deep learning fits: grading is image pattern recognition with large
  labelled datasets available; landmark result = Gulshan et al. 2016 [1].
- Aims: (i) train and compare CNNs for 5-class grading on APTOS 2019,
  (ii) quantify the value of transfer learning and fundus-specific
  preprocessing, (iii) assess clinical plausibility of model attention with
  Grad-CAM, (iv) evaluate the binary referable-DR screening view.
- Scope: single public dataset, Colab-scale compute — stated explicitly.

## 2. Background / Literature Review (~450 words) — rubric: Understanding 20%
- Clinical grading: the 5-grade international scale; what lesions define each
  grade (microaneurysms → haemorrhages/exudates → neovascularisation).
- Prior work: Gulshan et al. 2016 (JAMA, Inception on 128k images) [1];
  Ting et al. 2017 (multi-ethnic validation) [2]; De Fauw et al. 2018
  (clinical deployment context, OCT) [3].
- Methods used here: transfer learning rationale; EfficientNet compound
  scaling (Tan & Le 2019) [4]; Grad-CAM (Selvaraju et al. 2017) [5];
  Ben Graham preprocessing from Kaggle DR competition reports [6].
- Positioning: those works used 100k+ images; this project asks what is
  achievable at APTOS scale (~3.7k) — that's why transfer learning is central.

## 3. Methodology (~700 words) — rubric: Methodology 25%
- **3.1 Dataset:** APTOS 2019, 3,662 images, grade distribution (bar chart,
  Fig. 1); class imbalance stated with numbers (~49% grade 0).
- **3.2 Preprocessing:** circle crop, resize to 320px cache, Ben Graham
  Gaussian-blur subtraction with equation and before/after figure (Fig. 2);
  justify: cross-camera illumination variance.
- **3.3 Splits:** stratified 70/15/15 train/val/test, frozen across all runs,
  seed 42; test set touched once per model.
- **3.4 Models:** ResNet-18 from scratch (baseline — isolates the value of
  transfer learning), EfficientNet-B0 and B3 (ImageNet init). Table of
  parameter counts + input sizes (Table 1). Architecture diagram (Fig. 3).
- **3.5 Training:** class-weighted cross-entropy (state the weights formula),
  AdamW, cosine schedule, augmentation (flips/rotation — justified: fundus
  orientation is arbitrary), early stopping on val QWK.
- **3.6 Metrics:** define QWK and WHY it is the clinical standard (ordinal
  labels; grading 4 as 0 must cost more than as 3); per-class F1; referable-DR
  binary view (grades ≥2) with sensitivity/specificity/AUC — how screening
  programmes actually consume such models.
- Repo/Colab link + one paragraph on reproducibility (seeds, frozen split,
  configs saved per run).

## 4. Experiments and Results (~800 words) — rubric: Results 30%
- **4.1 Main comparison** (Table 2): QWK / accuracy / macro-F1 / referable
  metrics for the three models. Learning curves (Fig. 4).
- **4.2 Ablation** (Table 3): EfficientNet-B0 ben vs plain. State effect size
  on QWK, not just "better".
- **4.3 Error structure:** confusion matrices (Fig. 5). Expected pattern:
  confusion concentrated between adjacent grades (esp. 1↔2, 3↔4); quantify
  what fraction of errors are ±1 grade.
- **4.4 Referable-DR screening view:** sensitivity/specificity at the 0.5
  operating point + AUC; compare against the BDA screening standard
  (≥80% sensitivity, ≥95% specificity) and against Gulshan et al.'s reported
  numbers — with the caveat that datasets differ.
- **4.5 Grad-CAM:** panels of correct predictions per grade (Fig. 6) and
  confident misclassifications (Fig. 7). Report what attention lands on.

## 5. Discussion (~450 words) — rubric: Results 30% + Originality 10%
- Interpret: how much did transfer learning buy (scratch vs B0)? Did B3's
  extra capacity/resolution help at this dataset size, or overfit?
  Did Ben Graham help, and on which grades?
- **The interesting bit:** cases where the prediction is right but attention
  is on artefacts/optic disc — what that means for clinical trust.
- Limitations: single dataset (population + camera bias — APTOS is from one
  Indian screening programme); label noise in DR grading (cite inter-grader
  variability from [1]); no external validation set.
- **Responsible AI (the brief asks for this explicitly):** fairness across
  populations/skin-pigmentation-linked fundus appearance; the cost asymmetry
  of false negatives (missed sight-threatening DR) vs false positives
  (referral burden); model as *triage assistant, not diagnostician*;
  explainability as a prerequisite for clinical acceptance; data privacy of
  medical images.
- Future work: ordinal regression head, ensembling, external validation on
  Messidor-2, test-time augmentation.

## 6. Conclusion (~150 words)
What was built, headline numbers, the one-sentence Grad-CAM finding, what it
would take to make this deployable.

## References (IEEE, not counted)
[1] Gulshan et al., "Development and Validation of a Deep Learning Algorithm
    for Detection of Diabetic Retinopathy in Retinal Fundus Photographs,"
    JAMA, vol. 316, no. 22, 2016.
[2] Ting et al., "Development and Validation of a Deep Learning System for
    Diabetic Retinopathy and Related Eye Diseases Using Retinal Images From
    Multiethnic Populations With Diabetes," JAMA, vol. 318, no. 22, 2017.
[3] De Fauw et al., "Clinically applicable deep learning for diagnosis and
    referral in retinal disease," Nature Medicine, vol. 24, 2018.
[4] M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks," ICML, 2019.
[5] Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization," ICCV, 2017.
[6] B. Graham, "Kaggle Diabetic Retinopathy Detection competition report,"
    2015. (competition winner's preprocessing writeup)
[7] APTOS 2019 Blindness Detection dataset, Kaggle, 2019.
(+ PyTorch/torchvision library citations)

---

## Figure/table checklist (all must be referenced in the text)
- [ ] Fig. 1 — class distribution bar chart
- [ ] Fig. 2 — raw vs crop vs Ben Graham, one image per grade
- [ ] Fig. 3 — architecture/pipeline diagram
- [ ] Fig. 4 — learning curves (loss + val QWK)
- [ ] Fig. 5 — confusion matrices (best model, counts + normalised)
- [ ] Fig. 6 — Grad-CAM correct predictions per grade
- [ ] Fig. 7 — Grad-CAM confident failures
- [ ] Table 1 — model comparison (params, input size, init)
- [ ] Table 2 — main results
- [ ] Table 3 — preprocessing ablation
