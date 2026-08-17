# Cardiovascular Disease Risk Classifier

ML Assignment 2 — BITS Pilani WILP, M.Tech (AIML/DSE), Machine Learning

## a. Problem Statement

Cardiovascular disease is diagnosed using a mix of demographic data, basic body
measurements, blood pressure, and lifestyle factors — information that's cheap
and non-invasive to collect compared to lab tests. This project builds and
compares five classification models that predict whether a patient has
cardiovascular disease (`cardio` = 1) or not (`cardio` = 0) from exactly that
kind of routine data, and packages the best-performing setup into an
interactive Streamlit app so predictions and metrics can be explored on new
patient data without touching code.

## b. Dataset Description

- **Source:** [Cardiovascular Disease dataset, Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (`cardio_train.csv`)
- **Raw size:** 70,000 rows, 11 predictive features + `id` + target
- **After cleaning:** 68,654 rows (1,346 rows dropped — 1.9% — for physiologically
  impossible values: negative/extreme blood pressure, `ap_hi < ap_lo`, and
  implausible height/weight; see notebook Part 2)
- **Target:** `cardio` — binary, near-perfectly balanced (50.0% / 50.0%)
- **Final feature set (13 features):** `gender`, `height`, `weight`, `ap_hi`
  (systolic BP), `ap_lo` (diastolic BP), `cholesterol`, `gluc` (glucose),
  `smoke`, `alco`, `active`, plus three engineered features — `age_years`,
  `bmi`, `pulse_pressure` — added because the raw feature count (11) fell
  short of the assignment's 12-feature minimum, and because raw `age` (stored
  in days) and separate height/weight are less informative than their
  derived, clinically standard equivalents.

## c. GitHub Repository Link

`<PASTE YOUR GITHUB REPO URL HERE AFTER YOU PUSH>`

## d. Models Used

All 5 models were trained on the same 80/20 stratified train/test split
(`random_state=42`) of the cleaned, scaled data.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7224 | 0.7884 | 0.7506 | 0.6571 | 0.7008 | 0.4474 |
| Decision Tree | 0.7278 | 0.7886 | 0.7390 | 0.6954 | 0.7166 | 0.4561 |
| kNN | 0.7222 | 0.7796 | 0.7338 | 0.6879 | 0.7101 | 0.4448 |
| Naive Bayes | 0.7138 | 0.7751 | 0.7652 | 0.6080 | 0.6776 | 0.4354 |
| Random Forest (Ensemble) | **0.7294** | **0.7981** | 0.7612 | 0.6602 | 0.7071 | **0.4621** |

*(Numbers are from an actual run of `train_models.py` / `train_models.ipynb`
on the cleaned dataset — rerun the notebook and these will reproduce exactly
given the fixed random seed, or shift slightly if you tune hyperparameters
further.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid, boring baseline. Second-highest precision (0.751) but the lowest recall (0.657) of the non-NB models — it's conservative about calling someone "diseased," which is exactly the wrong bias for a screening tool where missed cases matter more than false alarms. |
| Decision Tree | Best recall (0.695) among the top performers and second-best MCC. A single depth-8 tree captures the main threshold effects (e.g. `ap_hi` and `age_years` cutoffs) reasonably well, but it's also the model most likely to overfit further if `max_depth` were relaxed — worth flagging in any follow-up tuning. |
| kNN | Middling everywhere, and the slowest at inference since it has to search the full training set at prediction time. With `n_neighbors=15` it doesn't have any obvious pathology, but it also has no clear advantage over the tree-based models here to justify its cost. |
| Naive Bayes | Weakest model on every metric except precision, where it's actually the *best* (0.765) — it's cautious. This tracks with theory: cholesterol, glucose, and blood-pressure readings are correlated with each other, and Naive Bayes assumes they aren't, which suppresses its ability to catch true positives (recall = 0.608, clearly the worst). |
| Random Forest (Ensemble) | Best AUC (0.798) and best MCC (0.462) — the most reliable ranker and the most balanced call overall, consistent with ensembling averaging away the overfitting risk a single Decision Tree carries. Its precision/recall trade-off sits between Logistic Regression and Decision Tree rather than dominating both, so the win here is about *consistency* across metrics, not a blowout on any single one. |
| **Overall Winner for this dataset** | **Random Forest** — best AUC and best MCC (the two metrics least distorted by threshold choice or class balance), and never the worst performer on any individual metric. Decision Tree is the closest runner-up and would be the pick if inference speed/interpretability mattered more than the last 1-2 points of MCC. |

**Why the numbers are close together (0.71–0.73 accuracy across the board):**
unlike a dataset with a few dominant, near-deterministic features, cardiovascular
risk here is genuinely diffuse across many moderately-informative features —
no single cutoff (e.g. "high cholesterol") reliably separates the classes, so
every model converges toward a similar ceiling. This is itself a legitimate
finding to note if you're asked about it.

## Repository Structure

```
cardio-classifier/
├── app.py                  # Streamlit app
├── train_models.py         # training script (plain .py version)
├── train_models.ipynb      # training notebook — RUN THIS ON BITS VIRTUAL LAB
├── requirements.txt
├── README.md
├── test_data.csv           # 2,000-row sample of the unscaled test set, for app upload
├── model_comparison.csv    # metrics table, machine-readable
├── confusion_matrices.png
├── .streamlit/config.toml  # app theme
└── model/
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## Streamlit App Features

- CSV upload (sidebar) — use the included `test_data.csv`
- Model selection dropdown (all 5 models)
- Live evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) if the
  uploaded CSV includes the `cardio` label column
- Confusion matrix + full classification report
- Row-level prediction table
- Side-by-side comparison of all 5 models on the uploaded data

## How to Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 train_models.py      # regenerates model/ and test_data.csv
streamlit run app.py
```

## Live App

`<PASTE YOUR STREAMLIT COMMUNITY CLOUD URL HERE AFTER DEPLOYING>`
