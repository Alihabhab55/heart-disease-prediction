# Heart Disease Prediction

An end-to-end machine-learning project that predicts the presence of heart disease from
routine clinical measurements, with an emphasis on **honest evaluation** and **decisions that
matter in a medical screening context**.

## Overview

The task is binary classification: given a patient's clinical profile (age, chest-pain type,
resting blood pressure, cholesterol, max heart rate, etc.), predict whether heart disease is
present.

What makes a *medical* screening problem different from a generic classification exercise is
the **asymmetry of errors**. Missing a patient who actually has disease (a false negative) can
be life-threatening, while a false alarm (a false positive) typically leads to further, less
costly testing. For that reason this project deliberately optimises for **recall** (sensitivity)
rather than raw accuracy, and tunes the decision threshold accordingly.

## Dataset

- **Source:** UCI Cleveland heart-disease data, via the Kaggle
  [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
  (the 1,025-row version). 13 clinical features + a binary `target`.
- The raw CSV is **not committed** to this repository. To reproduce, download `heart.csv`
  from the Kaggle link above and place it in the project root (see [How to run](#how-to-run)).

> **This is a learning project, not a clinical tool.** It uses a small, decades-old research
> dataset and is intended to demonstrate an ML workflow and sound reasoning — not to inform any
> real medical decision. Do not use it for diagnosis.

## Key decisions

These are the judgment calls that shaped the project — and where most of the real work was:

- **Removed 723 duplicate rows that would have leaked the test set into training.**
  The 1,025-row file contains only **302 unique patients**; the rest are exact duplicates.
  Left in place, identical rows would appear in both training and test splits, inflating the
  score and producing a model that looks great but hasn't generalised. Dropping them
  (1,025 → 302) is what makes every number below trustworthy.

- **Caught and corrected reverse-encoded labels by reasoning from cardiac physiology.**
  Grouping by the raw `target` showed the supposedly *diseased* group had a *healthier*
  profile — higher max heart rate (158 vs 139 bpm), lower ST depression (`oldpeak`), fewer
  major vessels affected (`ca`), and less exercise-induced angina (`exang`). That is
  backwards, so the labels were inverted (`target = 1 - target`) before modelling.

- **Checked for physiologically impossible values.** Screened for impossible readings such as
  resting blood pressure or cholesterol of 0. In this cleaned version of the data none were
  present, so no rows were removed on this basis — but the check is part of the pipeline.

- **Prioritised recall over accuracy and tuned the decision threshold.** Lowering the
  classification threshold from 0.50 to 0.30 raised recall from 0.75 to **0.93**, cutting
  missed cases (false negatives) on the test set from **7 to 2** — accepting more false alarms
  (6 → 9) as a deliberate, context-appropriate trade-off.

- **Reported an honest, cross-validated estimate.** Rather than quoting a single lucky split,
  performance is reported as 5-fold cross-validated ROC-AUC **with its spread**
  (~0.91 ± 0.02).

- **Confirmed the model relies on clinically meaningful features.** The most important
  predictors (`ca`, `cp`, `oldpeak`, `thalach`) are consistent across the random forest, the
  logistic-regression coefficients, the exploratory correlations, and known cardiology.

## Methodology

1. **Exploratory data analysis** — distributions, disease rates by feature, and a correlation
   analysis of each feature against the target.
2. **Preprocessing** — remove duplicates, correct the reversed labels, check for impossible
   values, then a **stratified** train/test split (80/20) and feature scaling (fit on the
   training data only, to avoid leakage).
3. **Logistic regression baseline** — a simple, interpretable reference model.
4. **Random forest** — a non-linear ensemble for comparison and for feature importances.
5. **Cross-validation** — 5-fold stratified CV (scaling done inside each fold) for an honest
   performance estimate.
6. **Threshold tuning** — adjust the decision threshold to favour recall for screening.
7. **Interpretation** — feature importances cross-checked against the EDA and cardiology.

## Results

On the held-out test set (61 patients):

| Model | Accuracy | Recall | Precision | ROC-AUC |
|-------|:--------:|:------:|:---------:|:-------:|
| Logistic Regression | 0.79 | 0.75 | 0.78 | 0.89 |
| Random Forest | 0.84 | 0.86 | 0.80 | 0.88 |

**Cross-validated ROC-AUC (5-fold):** Logistic Regression **0.905 ± 0.018**,
Random Forest **0.908 ± 0.016** — i.e. roughly **0.91**, with a tight spread.

**Threshold tuning (logistic regression).** The interpretable model was chosen for the
operating-point analysis:

| Threshold | Recall | Precision | Missed cases (FN) | False alarms (FP) |
|:---------:|:------:|:---------:|:-----------------:|:-----------------:|
| 0.50 (default) | 0.75 | 0.78 | 7 | 6 |
| 0.30 (chosen)  | 0.93 | 0.74 | **2** | 9 |

For a screening tool, cutting missed cases from 7 to 2 is well worth a few extra false alarms.

**Top features (random forest importance):** `ca` (number of major vessels), `cp`
(chest-pain type), `oldpeak` (exercise-induced ST depression), and `thalach` (max heart rate)
— all clinically plausible and consistent with the exploratory analysis.

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Get heart.csv from the Kaggle link in the Dataset section above
# and place it in the project root (next to the notebook).

# 5. Launch the notebook
jupyter notebook heart_disease_prediction.ipynb
```

Then run the cells top to bottom.

## Limitations

- **Small dataset.** After de-duplication only ~302 patients remain, so metrics carry real
  uncertainty — hence cross-validation and reporting the spread rather than a single number.
- **Old and narrow data.** The Cleveland data is decades old and not representative of the
  general population; a model trained on it would not transfer to today's clinical practice.
- **Not validated for clinical use.** No external validation, calibration analysis, or
  fairness/subgroup auditing has been done. This is a portfolio/learning project only.
- **Feature importance ≠ causation.** Important features are predictive in this dataset; that
  is not evidence of a causal clinical relationship.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn
