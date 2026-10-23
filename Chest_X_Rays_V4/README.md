# Chest X-Ray Multi-Condition Detection — Classical ML Pipeline
DataML – ML Value Challenge

This project predicts whether 5 conditions are present in a chest X-ray:

- Atelectasis
- Effusion
- Infiltration
- Nodule
- Pneumothorax

An X-ray can show more than one condition at once, so this is multi-label classification — 5 separate yes/no questions per image, not one "sick vs healthy" question.

Rule followed: no deep learning, no CNNs, no pretrained models. Every prediction comes from hand-crafted image features + classical ML models (Logistic Regression, HistGradientBoosting, SVM, ExtraTrees).

## What's new in this version

Changes since the previous notebook (feature set G):

- **New feature set H** — adds a 17th family, vessel/rib density, on top of feature set G. `feature_set` is now set to `"H"`.
- **Vessel/rib density features (new):** a Frangi vesselness filter approximates the density of linear structures (blood vessels, bronchial markings, ribs) — a different signal family from the existing texture-roughness features (LBP/GLCM/Gabor). Computed overall and separately for left/right lung, giving a vessel left-right ratio.
- **Feature-space outlier detection (new):** an IsolationForest flags images whose extracted features look anomalous, as a second line of defense alongside the existing image-level QC (corrupt files, duplicates). Cross-checked against the existing `segmentation_ok` flag to see how much overlap there is. An ablation compares OOF performance with and without the flagged rows in training.
- **Classifier Chain ensemble (new):** models each condition using the other conditions' predictions as extra input features, so the co-occurrence patterns already seen in the EDA get used by the model, not just documented. Averaged over 5 random label orderings to avoid depending on one arbitrary chain order.
- **Fourth ensembling method:** the per-condition automatic ensemble selection now compares weighted average, stacked meta-model, rank average, **and** classifier chain, and keeps whichever wins OOF for each condition.
- **Split conformal prediction (new):** instead of only reporting a raw probability, this adds a coverage guarantee — "the true label falls in this predicted set at least 90% of the time" — calibrated on OOF data. Cases where the interval can't commit to positive or negative are flagged `uncertain`, both for OOF and for the test set, which gives a concrete "% of cases that need human review" number.
- **SHAP interpretability (new):** SHAP summary plots per condition on the final ExtraTrees models, as a second, complementary lens to the mutual-information and permutation-importance views already in the notebook.
- **Inference latency benchmark (new):** measures milliseconds-per-image for the full feature set (H) versus a cheaper subset (feature set C, dropping the slowest families — wavelet, Gabor, GLRLM), to report a speed/accuracy tradeoff rather than a single latency number.
- **Composite asymmetry-severity index (new):** combines the existing symmetry, vessel-ratio, and lung-shape features into one interpretable number, and checks whether it tracks how many conditions a patient has at once, not just which single condition.
- The diagnostic fingerprint radar chart now also reflects vessel/rib density, since it's computed over whichever feature families are present in the active feature set (H).


## 1. Overall Pipeline

```
Raw X-ray images
  → Load & clean data (drop corrupt/missing images, find duplicates)
  → Preprocess each image (fix orientation, crop borders, normalize, enhance contrast)
  → Extract lung region (rough segmentation)
  → Extract hand-crafted features (feature set H: 17 families, incl. vessel/rib density)
  → Clean features (remove useless/redundant ones)
  → Feature-space outlier check (IsolationForest, cross-checked against segmentation QC)
  → Group-safe 5-fold split (duplicate images never span folds)
  → Tune hyperparameters per model (RandomizedSearchCV)
  → Train 4 classical models per condition, inside a leak-proof pipeline (SMOTE on SVM branch only)
  → Calibrate probabilities (configurable method/cv)
  → Combine 4 models per condition four ways: weighted average, stacked meta-model, rank average, classifier chain
  → Auto-select the best-performing ensemble method per condition
  → Calibrate conformal-prediction thresholds for a 90% coverage guarantee
  → Retrain everything on full training data
  → Predict on test images → predictions.csv (with per-condition "uncertain" flags)
```

Built and run in Google Colab, using data stored on Google Drive.

## 2. Data Handling

- Auto-locates the labels CSV and image folders inside the extracted zip.
- Drops corrupt/unreadable images, logs them separately rather than silently skipping.
- Finds duplicate/near-duplicate images via perceptual hashing, so they don't leak across folds.

## 3. Image Preprocessing

Every image goes through the same steps: polarity correction (fixes inverted scans), black-border cropping, aspect-ratio-preserving resize to 512×512, intensity normalization, and CLAHE/gamma contrast enhancement for the texture-based features. Each image ends up with a plain ("base") and an enhanced ("texture") version.

## 4. Lung Region Detection

Classical segmentation (Otsu thresholding + morphology) approximates the lung field, used for the body midline (left/right comparisons), ROI-only features, the clinical costophrenic-angle and lung-shape features, and — this version — the left/right split for vessel-density features. A `segmentation_ok` flag flags unreliable segmentations rather than hiding them.

## 5. Feature Extraction — Feature Set H (17 families)

| Family | What it measures |
|---|---|
| Intensity | Brightness, contrast, spread of pixel values |
| Texture (LBP + GLCM) | Local patterns/roughness |
| Gradient/Edges | Sharpness and location of edges |
| Morphology | Shapes, blobs, bright/dark regions |
| Spatial regions | Grid + top/middle/bottom zone stats |
| Symmetry | Left lung vs right lung comparison |
| Wavelet | Fine detail at multiple zoom levels |
| Frequency (FFT) | Fine detail vs smooth areas |
| Blob/lesion | Small round spot-like structures |
| ROI-aware | Same stats, restricted to the lung mask |
| GLRLM | Run-length of similar intensity — proxy for nodular texture |
| Gabor | Multi-frequency, multi-orientation texture response |
| Hu moments | Rotation/scale/translation-invariant shape descriptors of the lung mask |
| Costophrenic angle | How sharply the lung field meets the diaphragm — blunted in pleural effusion, computed per lung |
| Lung shape/volume | Left-right area/height/aspect ratios of the lung mask — sensitive to collapse and diaphragm flattening |
| **Vessel/rib density (new)** | Frangi-filter density of vessel-like linear structures, overall and left vs right |

The vessel/rib density family is what's new in feature set H (H = G + vessel).

## 6. Feature Cleaning

Drops near-constant features (VarianceThreshold) and highly correlated duplicates (>0.95 correlation).

## 6b. Feature-Space Outlier Check (new)

An IsolationForest is fit on the cleaned feature matrix to flag images that loaded fine but produced anomalous feature values — a second, independent line of defense on top of the image-level QC in Section 2. The overlap with the existing `segmentation_ok` flag is reported, and an ablation compares OOF macro AUC with and without the flagged rows kept in training. Flagged rows are excluded only from training, never from test, since real-world test images can't be discarded.

## 7. Validation Strategy (Leakage-Safe)

GroupKFold with duplicate-image groups: every duplicate/near-duplicate image is assigned a shared group id, so GroupKFold guarantees no group is split across folds.

## 8. Models

Four classical models trained per condition (One-vs-Rest):

- Logistic Regression (L1, tuned C)
- HistGradientBoosting (tuned depth/iterations/learning rate)
- SVM (RBF, tuned C/gamma)
- ExtraTreesClassifier (tuned n_estimators/min_samples_leaf)

Hyperparameters are tuned once via RandomizedSearchCV (per label, per model) and collapsed into a single global setting per model.

Each model's pipeline: mutual-information feature selection (~150 features, fit per training fold) → scaling → BorderlineSMOTE oversampling (SVM branch only by default) → classifier.

## 9. Calibration

CalibratedClassifierCV wraps every model so probabilities are trustworthy, not just directionally correct. Calibration method/cv are explicit parameters (defaulting to isotonic for logreg/histgb/extratrees, sigmoid for SVM). Calibration curves plot predicted vs. observed frequency per condition.

## 10. Ensemble

Four methods are computed per condition, and the best one for that condition is kept:

- **Weighted average** — per-condition weights optimized to minimize log loss.
- **Stacked meta-model** — a Logistic Regression trained on the 4 base models' OOF probabilities, validated with its own CV.
- **Rank average** — averages each model's rank (not raw probability) per condition, then rescales to [0, 1]; more robust when models disagree on calibration scale.
- **Classifier chain (new)** — models each condition using the other conditions' predictions as additional input features, averaged over 5 random label orderings so the result doesn't depend on one arbitrary chain order.

All four are compared using OOF ROC-AUC, and the winning method is selected per condition rather than globally. Rank averaging is OOF-only and falls back to the weighted average at test time, since ranks depend on which batch of images is scored together.

## 11. Evaluation

Per-condition and macro-averaged: ROC-AUC, PR-AUC, log loss, Brier score.

## 12. Conformal Prediction (new)

For each condition, OOF probabilities are used to calibrate a threshold pair that gives a 90% coverage guarantee: the true label falls inside the predicted set (positive / negative / uncertain) at least 90% of the time. This is applied to both OOF and test predictions, and the fraction of test cases landing in "uncertain" is reported per condition — a concrete number for how much of the test set would need human review at that confidence level.

## 13. Final Model + Test Predictions

All 4 models retrained on full training data, the per-condition ensemble method/weights from validation reused, and the same preprocessing/feature extraction applied to test images (no re-fitting on test data). Predictions saved to `predictions.csv`, including per-condition `uncertain` flags from the conformal thresholds. Sanity checks compare prediction averages to real prevalence, and check test-vs-train feature distribution shift (KS test).

## 14. Interpretability (expanded)

- SHAP summary plots per condition (new), on the final ExtraTrees models.
- Permutation importance per condition, on the final ExtraTrees models.
- Mutual-information feature-family radar chart ("diagnostic fingerprint", carried over from earlier versions) — which feature family (texture, symmetry, vessel density, etc.) each condition depends on most; now also covers vessel/rib density since that family is part of feature set H.
- Mutual-information top-10-features bar chart and boxplots per condition.
- PCA view of how visually distinct each condition is from a normal scan.
- Composite asymmetry-severity index (new) combining symmetry, vessel-ratio, and lung-shape features, checked against how many conditions a patient has at once.

## 15. Extra Insight / Wildcard Visualizations

All carried over unchanged from earlier notebook versions:

- Condition co-occurrence, as a heatmap and as a Jaccard-weighted network graph.
- Conditions-per-patient distribution, and which conditions drive multi-condition cases.
- Left-right symmetry breakdown (segmentation-based), plus a raw-pixel laterality check as a cross-check.
- Raw-pixel brightness-shift meter per condition vs. normal scans.
- Clinical urgency/severity gauge and clinical-priority pyramid — plain-language framings of prevalence and co-occurrence.
- Sample X-ray grid per condition for a visual sanity check.

## 16. Inference Latency Benchmark (new)

Reports milliseconds-per-image for the full feature set (H) versus a cheaper subset (feature set C — intensity/spatial/texture/gradient/morphology/symmetry only, dropping the slowest families: wavelet, Gabor, GLRLM). Intended to be read alongside the accuracy numbers from feature set C vs. H as a speed/accuracy tradeoff, not a standalone speed claim.

## 17. Tools & Libraries

OpenCV, scikit-image, SciPy, PyWavelets, scikit-learn, imbalanced-learn, iterative-stratification, NumPy, pandas, Matplotlib, Seaborn, NetworkX, Plotly/Kaleido, UpSetPlot, SHAP. No TensorFlow/PyTorch, no CNNs, no pretrained models.

## 18. How to Run

1. Open the notebook in Google Colab.
2. Run the setup cell to install packages and mount Google Drive.
3. Set `ZIP_FILENAME` to your dataset zip on Drive.
4. Run all cells top to bottom — CSV and image folders are auto-detected.
5. Extracted features are cached to Drive, so re-running won't recompute from scratch.
6. Final predictions are saved to `predictions.csv` in the configured output folder on Drive, including per-condition `uncertain` flags.
