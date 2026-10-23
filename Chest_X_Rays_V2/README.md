# Chest X-Ray Multi-Condition Detection — Classical ML Pipeline
DataML – ML Value Challenge

This project predicts whether 5 conditions are present in a chest X-ray:

- Atelectasis
- Effusion
- Infiltration
- Nodule
- Pneumothorax

An X-ray can show more than one condition at once, so this is multi-label classification — 5 separate yes/no questions per image, not one "sick vs healthy" question.

**Rule followed:** no deep learning, no CNNs, no pretrained models. Every prediction comes from hand-crafted image features + classical ML models (Logistic Regression, HistGradientBoosting, SVM, ExtraTrees).

---

## What's new in this version

Changes since the previous notebook:

- **3 new feature families** — GLRLM (run-length texture), Gabor filters (multi-orientation texture), and Hu moments (shape invariants of the lung mask). Combined into a new feature set, **F**, which is now the active setting.
- **4th model added** — ExtraTreesClassifier, alongside the existing Logistic Regression, HistGradientBoosting, and SVM.
- **Hyperparameter tuning** — RandomizedSearchCV is run per label per model, then collapsed into one tuned setting per model (median for numeric params, most common value for categorical ones) so `make_pipeline` stays simple.
- **Imbalance handling changed** — replaced `class_weight="balanced"` with BorderlineSMOTE oversampling inside the pipeline (fit only on the training fold, so it stays leakage-safe).
- **Bug fix** — removed the manual `sample_weight` for HistGradientBoosting. It broke under `CalibratedClassifierCV`'s internal fold splitting once SMOTE changed the row counts inside the pipeline.
- **CV grouping made stricter** — switched from `MultilabelStratifiedKFold` (which only warned if a duplicate group ended up split across folds) to `GroupKFold` using the duplicate-image groups directly. No duplicate/near-duplicate image can span two folds now — it's structural, not just monitored.
- **Second ensembling method added** — a stacked meta-model (Logistic Regression trained on the 4 base models' OOF probabilities), compared against the existing per-condition weighted-average ensemble.
- **Calibration curves added** — plots checking whether a predicted 0.7 is right about 70% of the time.
- **SMOTE ablation run** — see below.

### Experiment: SMOTE ablation

I compared OOF ROC-AUC with and without SMOTE, same folds, same models:

| Condition | Model | Without SMOTE | With SMOTE |
|---|---|---|---|
| Atelectasis | histgb | 0.673 | 0.647 |
| Effusion | histgb | 0.712 | 0.681 |
| Infiltration | histgb | 0.643 | 0.617 |
| Nodule | histgb | 0.606 | 0.580 |
| Pneumothorax | histgb | 0.663 | 0.685 |
| Atelectasis | svm | 0.618 | 0.629 |
| Nodule | svm | 0.558 | 0.600 |

Full pattern across all 4 models: SMOTE mostly **lowered** AUC for HistGB, ExtraTrees, and Logistic Regression, and mostly **raised** it for SVM. Net effect was negative for the tree-based and linear models, which carry more weight in the ensemble.

This isn't a final decision, just a documented finding — the pipeline still applies SMOTE by default (`use_smote=True`) for now. Next step is either restricting SMOTE to the SVM branch only, or dropping it in favor of the existing class-weighting.

---

## 1. Overall Pipeline

```
Raw X-ray images
  → Load & clean data (drop corrupt/missing images, find duplicates)
  → Preprocess each image (fix orientation, crop borders, normalize, enhance contrast)
  → Extract lung region (rough segmentation)
  → Extract hand-crafted features (feature set F: 15 families)
  → Clean features (remove useless/redundant ones)
  → Group-safe 5-fold split (duplicate images never span folds)
  → Tune hyperparameters per model (RandomizedSearchCV)
  → Train 4 classical models per condition, inside a leak-proof pipeline (with SMOTE)
  → Calibrate probabilities
  → Combine 4 models per condition two ways: weighted average, and stacked meta-model
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

Classical segmentation (Otsu thresholding + morphology) approximates the lung field, used for the body midline (left/right comparisons) and for ROI-only features. A `segmentation_ok` flag flags unreliable segmentations rather than hiding them.

## 5. Feature Extraction — Feature Set F (15 families)

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
| **GLRLM** *(new)* | Run-length of similar intensity — short high-intensity runs as a proxy for nodular texture |
| **Gabor** *(new)* | Multi-frequency, multi-orientation texture response |
| **Hu moments** *(new)* | Rotation/scale/translation-invariant shape descriptors of the lung mask |

## 6. Feature Cleaning

Drops near-constant features (`VarianceThreshold`) and highly correlated duplicates (>0.95 correlation).

## 7. Validation Strategy (Leakage-Safe)

`GroupKFold` with duplicate-image groups: every duplicate/near-duplicate image is assigned a shared group id, so `GroupKFold` guarantees no group is split across folds. This replaces the earlier stratified-fold-plus-warning approach with a hard guarantee.

## 8. Models

Four classical models trained per condition (One-vs-Rest):

- Logistic Regression (L1, tuned `C`)
- HistGradientBoosting (tuned depth/iterations/learning rate)
- SVM (RBF, tuned `C`/`gamma`)
- **ExtraTreesClassifier** *(new)* (tuned `n_estimators`/`min_samples_leaf`)

Hyperparameters are tuned once via `RandomizedSearchCV` (per label, per model) and collapsed into a single global setting per model.

Each model's pipeline: mutual-information feature selection (~150 features, fit per training fold) → scaling → BorderlineSMOTE oversampling (train fold only) → classifier.

## 9. Calibration

`CalibratedClassifierCV` wraps every model so probabilities are trustworthy, not just directionally correct. Calibration curves now plot predicted vs. observed frequency per condition.

## 10. Ensemble

Two methods are computed and compared per condition:

- **Weighted average** — per-condition weights optimized to minimize log loss.
- **Stacked meta-model** *(new)* — a Logistic Regression trained on the 4 base models' OOF probabilities, validated with its own CV.

Both are compared against each other and against the best single model.

## 11. Evaluation

Per-condition and macro-averaged: ROC-AUC, PR-AUC, log loss, Brier score.

## 12. Final Model + Test Predictions

All 4 models retrained on full training data, same ensemble weights reused, same preprocessing/feature extraction applied to test images (no re-fitting on test data). Predictions saved to `predictions.csv`. Sanity checks compare prediction averages to real prevalence, and check test-vs-train feature distribution shift (KS test).

## 13. Extra Insight Analyses

Exploratory visualizations for interpretability: condition co-occurrence, conditions-per-patient distribution, left-right symmetry breakdown, which feature family each condition relies on most (mutual information), and a PCA view of how visually distinct each condition is from a normal scan.

## 14. Tools & Libraries

OpenCV, scikit-image, SciPy, PyWavelets, scikit-learn, imbalanced-learn, iterative-stratification, NumPy, pandas, Matplotlib, Seaborn, NetworkX. No TensorFlow/PyTorch, no CNNs, no pretrained models.

## 15. How to Run

1. Open the notebook in Google Colab.
2. Run the setup cell to install packages and mount Google Drive.
3. Set `ZIP_FILENAME` to your dataset zip on Drive.
4. Run all cells top to bottom — CSV and image folders are auto-detected.
5. Extracted features are cached to Drive, so re-running won't recompute from scratch.
6. Final predictions are saved to `predictions.csv` in the configured output folder on Drive.
