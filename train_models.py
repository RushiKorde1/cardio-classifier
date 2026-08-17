"""
Cardiovascular Disease Classification - Model Training
ML Assignment 2 | BITS Pilani M.Tech AIML

Dataset: Kaggle Cardiovascular Disease dataset (cardio_train.csv)
Target: cardio (0 = no disease, 1 = disease) - binary classification
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("cardio_train.csv", sep=";")
print("Raw shape:", df.shape)

# ---------------------------------------------------------------
# 2. Clean bad / impossible entries
#    (this dataset has known data-entry errors: negative & absurd
#     blood pressure values, ap_hi < ap_lo, extreme height/weight)
# ---------------------------------------------------------------
before = len(df)
df = df[(df.ap_hi > 0) & (df.ap_hi <= 250)]
df = df[(df.ap_lo > 0) & (df.ap_lo <= 200)]
df = df[df.ap_hi >= df.ap_lo]
df = df[(df.height >= 120) & (df.height <= 220)]
df = df[(df.weight >= 30) & (df.weight <= 200)]
print(f"Dropped {before - len(df)} rows as physiologically invalid "
      f"({(before-len(df))/before:.1%} of data)")

# ---------------------------------------------------------------
# 3. Feature engineering
#    raw dataset only has 11 usable features (excl. id/target),
#    below the assignment's 12-feature minimum -> engineer 3 more
# ---------------------------------------------------------------
df["age_years"] = (df["age"] / 365.25).round(1)
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

df = df.drop(columns=["id", "age"])  # id is noise, raw age (days) replaced by age_years

feature_cols = [c for c in df.columns if c != "cardio"]
print(f"\nFinal feature count: {len(feature_cols)}")
print(feature_cols)
print(f"Final instance count: {len(df)}")

# ---------------------------------------------------------------
# 4. Train / test split (stratified) + scaling
# ---------------------------------------------------------------
X = df[feature_cols]
y = df["cardio"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only, never fit_transform on test

# ---------------------------------------------------------------
# 5. Define and train models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, random_state=RANDOM_STATE
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE
    ),
}

trained_models = {}
for name, m in models.items():
    m.fit(X_train_scaled, y_train)
    trained_models[name] = m
    print(f"Trained: {name}")

# ---------------------------------------------------------------
# 6. Evaluate: Accuracy, AUC, Precision, Recall, F1, MCC
# ---------------------------------------------------------------
results = []
cms = {}
for name, m in trained_models.items():
    y_pred = m.predict(X_test_scaled)
    y_proba = m.predict_proba(X_test_scaled)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })
    cms[name] = confusion_matrix(y_test, y_pred)

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n" + "=" * 70)
print(results_df.to_string())
print("=" * 70)

# ---------------------------------------------------------------
# 7. Confusion matrices grid
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for ax, (name, cm) in zip(axes, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
print("\nSaved confusion_matrices.png")

# ---------------------------------------------------------------
# 8. Save artifacts: models, scaler, test data, results table
# ---------------------------------------------------------------
os.makedirs("model", exist_ok=True)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

filename_map = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
}
for name, m in trained_models.items():
    with open(f"model/{filename_map[name]}.pkl", "wb") as f:
        pickle.dump(m, f)

with open("model/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# Save a sample of the (unscaled) test set for the Streamlit app upload demo
test_df = X_test.copy()
test_df["cardio"] = y_test.values
test_sample = test_df.sample(n=min(2000, len(test_df)), random_state=RANDOM_STATE)
test_sample.to_csv("test_data.csv", index=False)
print(f"\nSaved test_data.csv with {len(test_sample)} rows")

results_df.to_csv("model_comparison.csv")
print("Saved model_comparison.csv")
print("\nDONE.")
