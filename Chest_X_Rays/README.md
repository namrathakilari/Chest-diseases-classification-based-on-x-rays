# Chest X-Ray Multi-Condition Detection — Classical ML Pipeline

**DataML – ML Value Challenge**

This project builds a machine learning pipeline that looks at chest X-ray images and predicts whether **5 medical conditions** are present:

- Atelectasis
- Effusion
- Infiltration
- Nodule
- Pneumothorax

An X-ray can show more than one condition at the same time, so this is a **multi-label classification** problem (not just "sick vs healthy" — it's 5 separate yes/no questions per image).

**Important rule followed:** No deep learning, no CNNs, no pretrained models were used anywhere in this project. Every prediction comes from **classical machine learning** — hand-crafted image features + traditional ML models (Logistic Regression, HistGradientBoosting, SVM). This was a requirement of the challenge.

---

## 1. Overall Pipeline

The project follows one continuous pipeline, from raw image to final prediction:

```
Raw X-ray images
   → Load & clean data (drop corrupt/missing images, find duplicates)
   → Preprocess each image (fix orientation, crop borders, normalize, enhance contrast)
   → Extract lung region (rough segmentation)
   → Extract hand-crafted features (12 different feature "families")
   → Clean features (remove useless/redundant ones)
   → Split data safely for validation (5-fold, stratified, leakage-checked)
   → Train 3 classical models per condition, inside a leak-proof pipeline
   → Calibrate probabilities (make predicted percentages trustworthy)
   → Combine (ensemble) the 3 models per condition with learned weights
   → Retrain everything on full training data
   → Predict on test images → predictions.csv
```

Everything was built and run in **Google Colab**, using data stored on Google Drive.

---

## 2. Data Handling

- The dataset comes as a zip file of X-ray images plus a `train_labels.csv` file (image name → 5 condition labels).
- The notebook automatically finds the CSV and image folders inside the zip after extracting it (no need to manually type long folder paths).
- **Data cleaning steps:**
  - Every image is checked — if a file is missing or corrupted, it's dropped and logged separately (not silently ignored).
  - **Duplicate/near-duplicate detection:** Using a perceptual hash (a fingerprint of what the image looks like), the code finds X-rays that are the same or nearly the same. This matters because if a duplicate ends up in both the training and validation set, the model would "cheat" by memorizing it — this is called **data leakage**, and the pipeline actively guards against it.

---

## 3. Image Preprocessing

Before any features are extracted, every image goes through the same cleanup steps so they're all comparable:

1. **Polarity correction** – Some X-rays are accidentally "inverted" (like a photo negative). The code detects this and flips it back.
2. **Border cropping** – Removes black scanner borders around the actual X-ray.
3. **Resize** – All images are resized to a standard size (512×512) while keeping their original aspect ratio (padding added instead of stretching/distorting them).
4. **Intensity normalization** – Adjusts brightness/contrast range so all images are on the same scale.
5. **CLAHE + Gamma correction** – A contrast-enhancement technique that makes fine details (like small nodules) more visible, used specifically for the texture-based features.

Each image ends up with two versions: a plain **normalized** version and an **enhanced (texture)** version — different features use whichever version suits them best.

---

## 4. Lung Region Detection

A simple, classical method (thresholding + shape cleanup) is used to roughly outline where the lungs are in the image. This is used to:
- Find the **body midline**, so left-lung vs right-lung comparisons make sense.
- Create features that only look **inside the lung area**, ignoring the background.
- The code checks if this rough segmentation looks reasonable (`segmentation_ok`) and reports how often it works, so this isn't a hidden weak point.

---

## 5. Feature Extraction — 12 Families

Since no neural network is allowed to "look at" the image and learn patterns on its own, the project instead calculates **hundreds of hand-designed numeric measurements** per image that describe what the image looks like. These are grouped into families:

| Family | What it measures |
|---|---|
| **Intensity** | Overall brightness, contrast, spread of pixel values |
| **Texture (LBP + GLCM)** | Local patterns/roughness in the image |
| **Gradient/Edges** | How sharp and where the edges/outlines are |
| **Morphology** | Shapes, blobs, bright/dark regions |
| **Spatial regions** | Splits the image into a grid (and top/middle/bottom zones) and measures each part separately |
| **Symmetry** | Compares the left lung to the right lung — diseases often make X-rays less symmetric |
| **Wavelet** | Fine detail at multiple zoom levels |
| **Frequency (FFT)** | How much "fine detail" vs "smooth areas" the image has |
| **Blob/lesion detection** | Looks for small round spot-like structures (useful for nodules) |
| **ROI-aware features** | Same stats as above, but calculated only inside the detected lung area |

A `feature_set` setting (A–E) controls how many of these families are used at once, so it's easy to test smaller vs bigger feature sets.

---

## 6. Feature Cleaning

Once all features are calculated:
- Features that barely change across images (near-constant, not useful) are dropped.
- Features that are almost identical to each other (highly correlated, redundant) are dropped, keeping only one.

This shrinks the feature set without losing information.

---

## 7. Validation Strategy (Leakage-Safe)

- Uses **Multilabel Stratified 5-Fold Cross-Validation** — this makes sure each of the 5 conditions is fairly represented in every fold, even though some conditions are rare.
- Checks whether any duplicate images accidentally landed in two different folds (which would leak information).
- All feature selection, scaling, and model fitting happens **inside** the training fold only — never touching validation data — so evaluation scores are trustworthy.

---

## 8. Models

For **each of the 5 conditions separately**, 3 different classical models are trained:

1. **Logistic Regression** (with L1 regularization, balanced for imbalance)
2. **HistGradientBoosting** (a tree-based boosting model)
3. **SVM** (Support Vector Machine)

This "one model per condition, per algorithm" setup is called **One-vs-Rest (OvR)** — it treats each disease as its own independent detection problem.

Each model is wrapped in a pipeline that does, in order:
1. Select the most useful ~150 features (using mutual information, calculated fresh on each training fold — no leakage).
2. Scale the features.
3. Train the classifier (with class weighting to handle the fact that most X-rays are "normal" for any given condition).

---

## 9. Calibration

Raw model outputs don't always mean what they say — a model might say "80% chance" when the real chance is closer to 60%. **Calibration** fixes this using `CalibratedClassifierCV`, so the final probabilities are more meaningful and trustworthy — useful since this is a medical-style problem where the confidence level matters, not just a yes/no answer.

---

## 10. Ensemble (Combining the 3 Models)

For each condition, the 3 models' predictions are combined using a **weighted average**, where the weights are automatically optimized (to minimize log loss) instead of guessed. Some conditions might rely more on the boosting model, others more on logistic regression — the weights adapt per condition.

The ensemble is compared against each individual model to confirm it actually performs better.

---

## 11. Evaluation

Model performance is measured per condition and averaged (macro) using:
- **ROC-AUC** – how well the model ranks positive vs negative cases
- **PR-AUC** – performance especially on rare/imbalanced conditions
- **Log Loss** and **Brier Score** – how well-calibrated the predicted probabilities are

These metrics were chosen as a general diagnostic panel since no single official leaderboard metric was specified.

---

## 12. Final Model + Test Predictions

After validation confirms the approach works:
- All 3 models (per condition) are **retrained on the full training dataset** (not just the training folds).
- The same ensemble weights from validation are reused.
- The exact same preprocessing + feature extraction is applied to the **test images** (no re-fitting on test data — it only uses what was already learned from training).
- Final predictions are saved to `predictions.csv`, with one probability column per condition.

A few sanity checks are also included afterward — comparing prediction averages to real-world prevalence, and checking whether test image features look statistically different from training features (distribution shift check).

---

## 13. Extra Insight Analyses

Beyond just building a predictor, the notebook also includes exploratory visualizations to explain *what the data and model are actually picking up on*, including:

- How often conditions occur together (co-occurrence heatmap)
- How many conditions a typical patient has at once
- Which conditions break left-right lung symmetry the most
- Which "family" of features (texture, shape, symmetry, etc.) each condition depends on most
- A 2D visual map (PCA) showing how visually distinct each condition is from a normal scan
- Sample X-ray images shown per condition for a visual sanity check

These are meant to add interpretability — showing *why* the model predicts what it predicts, not just the accuracy numbers.

---

## 14. Tools & Libraries Used

- **Image processing:** OpenCV, scikit-image, SciPy, PyWavelets
- **Machine learning:** scikit-learn, LightGBM/XGBoost (optional), iterative-stratification
- **Data handling:** NumPy, pandas
- **Visualization:** Matplotlib, Seaborn, NetworkX (for the co-occurrence network graph)
- **Environment:** Google Colab (with Google Drive for storage/caching)

No CNNs, no pretrained models, no deep learning frameworks (TensorFlow/PyTorch neural nets) were used — strictly classical ML, as required by the competition rules.

---

## 15. How to Run

1. Open the notebook in Google Colab.
2. Run the setup cell to install required packages and mount Google Drive.
3. Update `ZIP_FILENAME` in the notebook to point to your dataset zip on Drive.
4. Run all cells top to bottom — the notebook auto-detects the CSV and image folders.
5. Extracted features are cached to Drive, so re-running the notebook later won't recompute everything from scratch.
6. Final predictions are saved to `predictions.csv` inside the configured output folder on Drive.
