# Chest X-Ray Multi-Condition Detection — Classical ML Pipeline
**DataML – ML Value Challenge**

This project predicts whether 5 conditions are present in a chest X-ray:

- Atelectasis
- Effusion
- Infiltration
- Nodule
- Pneumothorax

An X-ray can show more than one condition at once, so this is **multi-label classification** — 5 separate yes/no questions per image, not one "sick vs healthy" question.

**Rule followed:** no deep learning, no CNNs, no pretrained models. Every prediction comes from hand-crafted image features + classical ML models (Logistic Regression, HistGradientBoosting, SVM, ExtraTrees).

## What's new in this version

Changes since the previous notebook:

- **New feature set G** — adds a 16th family, **clinical/anatomical features** (costophrenic-angle sharpness + gross lung-shape/volume ratios), on top of feature set F. `feature_set` is now set to `"G"`.
  - *Costophrenic angle sharpness*: approximates how quickly the lung mask narrows near the diaphragm — a normal chest X-ray meets the diaphragm in a sharp point, while pleural effusion "blunts" that corner with fluid. Computed separately per lung since effusion is often one-sided.
  - *Lung shape/volume*: left-vs-right area ratio, height ratio, and aspect ratio from the lung mask — atelectasis (lung collapse) shrinks and distorts one lung, and effusion can flatten the diaphragm line.
- **SMOTE ablation resolved** — following the previous version's finding that SMOTE mostly hurt the tree-based and linear models but helped SVM, oversampling is now applied **only to the SVM branch** by default (`use_smote=None` auto-resolves to SVM-only), with an explicit override still available for re-running the ablation.
- **Third ensembling method — rank averaging** — in addition to the per-condition weighted average and the stacked meta-model, predictions are now also combined by averaging each model's *rank* (not raw probability) per condition, then rescaling to [0, 1]. This is more robust when models disagree on calibration scale, which raw-probability averaging doesn't always resolve across very different model families (SVM vs. tree-based).
- **Automatic per-label ensemble selection** — instead of picking one ensembling strategy for every condition, the notebook now compares weighted-average, stacked, and rank-average OOF ROC-AUC **per condition** and keeps whichever wins for that specific condition, since different conditions responded differently to each ensembling style.
- **Permutation-importance interpretability** — added a feature-importance report (`top_features_per_label`) using `sklearn.inspection.permutation_importance` on the final ExtraTrees pipeline per condition, mapped back to human-readable feature names, as a second lens on top of the existing mutual-information feature-family analysis.
- **Calibration made configurable** — `make_calibrated_pipeline` now accepts explicit `calib_method`/`calib_cv` so calibration choices (isotonic vs. sigmoid, number of folds) can be swept and compared, while defaulting to the original behavior (isotonic for logreg/histgb, sigmoid for SVM) so existing results stay reproducible.

The exploratory "wildcard" analyses (raw-pixel brightness-shift meter, raw-pixel left-right laterality check, and the NetworkX co-occurrence graph) carry over unchanged from the earlier weekly wildcard-challenge submissions — see Section 13 below for what they do.

This isn't a final state — the per-condition ensemble auto-selection and the new clinical feature family are the two things most likely to be tuned further next.

## 1. Overall Pipeline

```
Raw X-ray images
  → Load & clean data (drop corrupt/missing images, find duplicates)
  → Preprocess each image (fix orientation, crop borders, normalize, enhance contrast)
  → Extract lung region (rough segmentation)
  → Extract hand-crafted features (feature set G: 16 families, incl. clinical/anatomical)
  → Clean features (remove useless/redundant ones)
  → Group-safe 5-fold split (duplicate images never span folds)
  → Tune hyperparameters per model (RandomizedSearchCV)
  → Train 4 classical models per condition, inside a leak-proof pipeline (SMOTE on SVM branch only)
  → Calibrate probabilities (configurable method/cv)
  → Combine 4 models per condition three ways: weighted average, stacked meta-model, rank average
  → Auto-select the best-performing ensemble method per condition
  → Retrain everything on full training data
  → Predict on test images → predictions.csv
```

Built and run in Google Colab, using data stored on Google Drive.

## 2. Data Handling

- Auto-locates the labels CSV and image folders inside the extracted zip.
- Drops corrupt/unreadable images, logs them separately rather than silently skipping.
- Finds duplicate/near-duplicate images via perceptual hashing, so they don't leak across folds.

## 3. Image Preprocessing

Every image goes through the same steps: polarity correction (fixes inverted scans), black-border cropping, aspect-ratio-preserving resize to 512×512, intensity normalization, and CLAHE/gamma contrast enhancement for the texture-based features. Each image ends up with a plain ("base") and an enhanced ("texture") version.

## 4. Lung Region Detection

Classical segmentation (Otsu thresholding + morphology) approximates the lung field, used for the body midline (left/right comparisons), ROI-only features, and — new this version — the clinical costophrenic-angle and lung-shape features. A `segmentation_ok` flag flags unreliable segmentations rather than hiding them.

## 5. Feature Extraction — Feature Set G (16 families)

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
| GLRLM | Run-length of similar intensity — short high-intensity runs as a proxy for nodular texture |
| Gabor | Multi-frequency, multi-orientation texture response |
| Hu moments | Rotation/scale/translation-invariant shape descriptors of the lung mask |
| **Costophrenic angle (new)** | How sharply the lung field meets the diaphragm — blunted in pleural effusion, computed per lung |
| **Lung shape/volume (new)** | Left-right area/height/aspect ratios of the lung mask — sensitive to collapse (atelectasis) and diaphragm flattening (effusion) |

The costophrenic-angle and lung-shape families are grouped together as the **"clinical"** family, which is what's new in feature set `G` (`G = F + clinical`).

## 6. Feature Cleaning

Drops near-constant features (`VarianceThreshold`) and highly correlated duplicates (>0.95 correlation).

## 7. Validation Strategy (Leakage-Safe)

`GroupKFold` with duplicate-image groups: every duplicate/near-duplicate image is assigned a shared group id, so `GroupKFold` guarantees no group is split across folds.

## 8. Models

Four classical models trained per condition (One-vs-Rest):

- Logistic Regression (L1, tuned C)
- HistGradientBoosting (tuned depth/iterations/learning rate)
- SVM (RBF, tuned C/gamma)
- ExtraTreesClassifier (tuned n_estimators/min_samples_leaf)

Hyperparameters are tuned once via `RandomizedSearchCV` (per label, per model) and collapsed into a single global setting per model.

Each model's pipeline: mutual-information feature selection (~150 features, fit per training fold) → scaling → BorderlineSMOTE oversampling (**SVM branch only by default**, based on the prior ablation) → classifier.

## 9. Calibration

`CalibratedClassifierCV` wraps every model so probabilities are trustworthy, not just directionally correct. Calibration method/cv are now explicit parameters (defaulting to isotonic for logreg/histgb/extratrees, sigmoid for SVM), so they can be swept in diagnostics without touching the main training path. Calibration curves plot predicted vs. observed frequency per condition.

## 10. Ensemble

Three methods are computed per condition, and the best one for that condition is kept:

- **Weighted average** — per-condition weights optimized to minimize log loss.
- **Stacked meta-model** — a Logistic Regression trained on the 4 base models' OOF probabilities, validated with its own CV.
- **Rank average (new)** — averages each model's rank (not raw probability) per condition, then rescales to [0, 1]; more robust when models disagree on calibration scale.

All three are compared against each other and against the best single model using OOF ROC-AUC, and the winning method is selected **per condition** rather than globally.

## 11. Evaluation

Per-condition and macro-averaged: ROC-AUC, PR-AUC, log loss, Brier score.

## 12. Final Model + Test Predictions

All 4 models retrained on full training data, the per-condition ensemble method/weights from validation reused, and the same preprocessing/feature extraction applied to test images (no re-fitting on test data). Predictions saved to `predictions.csv`. Sanity checks compare prediction averages to real prevalence, and check test-vs-train feature distribution shift (KS test).

## 13. Extra Insight Analyses

Exploratory visualizations for interpretability:

- Condition co-occurrence, both as a heatmap and as a **network graph** (NetworkX, Jaccard-weighted edges) — carried over from the earlier wildcard-challenge submissions.
- Conditions-per-patient distribution.
- Left-right symmetry breakdown (segmentation-based), plus a raw-pixel laterality check (also from the wildcard submissions) as a cross-check that the segmentation-based symmetry signal isn't an artifact.
- A raw-pixel brightness-shift meter per condition vs. normal scans (wildcard submission), as a quick sanity signal before trusting the hand-crafted features.
- Which feature family each condition relies on most (mutual information), plus a **permutation-importance** view (new this version) on the final ExtraTrees models.
- A PCA view of how visually distinct each condition is from a normal scan.

## 14. Tools & Libraries

OpenCV, scikit-image, SciPy, PyWavelets, scikit-learn, imbalanced-learn, iterative-stratification, NumPy, pandas, Matplotlib, Seaborn, NetworkX, Plotly/Kaleido, UpSetPlot. No TensorFlow/PyTorch, no CNNs, no pretrained models.

## 15. How to Run

1. Open the notebook in Google Colab.
2. Run the setup cell to install packages and mount Google Drive.
3. Set `ZIP_FILENAME` to your dataset zip on Drive.
4. Run all cells top to bottom — CSV and image folders are auto-detected.
5. Extracted features are cached to Drive, so re-running won't recompute from scratch.
6. Final predictions are saved to `predictions.csv` in the configured output folder on Drive.
