"""
Cardiovascular Disease Risk Classifier - Streamlit App
ML Assignment 2 | BITS Pilani M.Tech AIML
"""
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Cardio Risk Classifier",
    page_icon="🫀",
    layout="wide",
)

# ---------------------------------------------------------------
# Styling — clinical chart aesthetic: deep slate + teal + a single
# coral accent reserved for "at risk" states, monospace for numbers
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0F1B22; color: #E7EEF0; }

.ecg-header {
    display: flex; align-items: center; gap: 16px;
    border-bottom: 1px solid #23343B; padding-bottom: 18px; margin-bottom: 24px;
}
.ecg-title { font-size: 1.9rem; font-weight: 700; color: #E7EEF0; margin: 0; }
.ecg-sub { color: #6FA8A0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin: 0; }

div[data-testid="stMetric"] {
    background-color: #17262C; border: 1px solid #23343B;
    border-radius: 6px; padding: 14px 16px;
}
div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #2DD4BF; }
div[data-testid="stMetricLabel"] { color: #8FA6AA; }

.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; }
.risk-pill {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 600;
}
.risk-high { background: rgba(232,85,61,0.15); color: #E8553D; border: 1px solid #E8553D; }
.risk-low { background: rgba(45,212,191,0.15); color: #2DD4BF; border: 1px solid #2DD4BF; }
</style>

<div class="ecg-header">
    <svg width="46" height="46" viewBox="0 0 46 46">
        <polyline points="2,23 12,23 16,10 20,36 24,16 28,30 32,23 44,23"
            fill="none" stroke="#2DD4BF" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
    </svg>
    <div>
        <p class="ecg-title">Cardiovascular Risk Classifier</p>
        <p class="ecg-sub">5 models trained on 68,654 patient records · cardio_train.csv (Kaggle)</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Load artifacts (cached so pickles load once, not on every click)
# ---------------------------------------------------------------
FILENAME_MAP = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
}

@st.cache_resource
def load_artifacts():
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    models = {}
    for name, fname in FILENAME_MAP.items():
        with open(f"model/{fname}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    return scaler, feature_cols, models

try:
    scaler, feature_cols, models = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `train_models.py` (or the notebook) "
        "first so the `model/` folder is populated, then redeploy."
    )
    st.stop()

# ---------------------------------------------------------------
# Sidebar — model selection + upload
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### Configuration")
    selected_model_name = st.selectbox("Model", list(models.keys()), index=4)
    st.markdown("---")
    st.markdown("### Test data")
    st.caption("Upload the provided `test_data.csv` (or any CSV with the same columns).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("---")
    st.caption(
        "Expected columns:\n\n" + ", ".join(feature_cols) + ", `cardio` (optional, for scoring)"
    )

if uploaded_file is None:
    st.info("Upload a CSV in the sidebar to see predictions and evaluation metrics. "
            "A ready-made `test_data.csv` is included in the repo.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

missing_cols = [c for c in feature_cols if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded file is missing required columns: {missing_cols}")
    st.stop()

has_labels = "cardio" in data.columns
model = models[selected_model_name]

X_input = data[feature_cols]
X_scaled = scaler.transform(X_input)
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Evaluation Metrics", "🧾 Predictions", "📈 Model Comparison"])

with tab1:
    st.markdown(f"**Model:** `{selected_model_name}`")
    if has_labels:
        y_true = data["cardio"].astype(int)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("AUC", f"{auc:.3f}")
        c3.metric("Precision", f"{prec:.3f}")
        c4.metric("Recall", f"{rec:.3f}")
        c5.metric("F1 Score", f"{f1:.3f}")
        c6.metric("MCC", f"{mcc:.3f}")

        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        fig.patch.set_facecolor("#0F1B22")
        ax.set_facecolor("#0F1B22")
        sns.heatmap(cm, annot=True, fmt="d", cmap="mako", ax=ax,
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"],
                    cbar=False, annot_kws={"color": "white"})
        ax.set_xlabel("Predicted", color="#E7EEF0")
        ax.set_ylabel("Actual", color="#E7EEF0")
        ax.tick_params(colors="#E7EEF0")
        st.pyplot(fig)

        st.markdown("#### Classification Report")
        report = classification_report(y_true, y_pred, target_names=["No Disease", "Disease"],
                                        output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    else:
        st.warning(
            "Uploaded CSV has no `cardio` column, so evaluation metrics can't be computed. "
            "Predictions are shown in the next tab."
        )

with tab2:
    st.markdown("#### Row-level predictions")
    out = data[feature_cols].copy()
    out["predicted_risk"] = np.where(y_pred == 1, "Disease", "No Disease")
    out["probability"] = y_proba.round(3)
    st.dataframe(out.head(200), use_container_width=True)
    st.caption(f"Showing first 200 of {len(out)} rows.")

with tab3:
    st.markdown("#### All 5 models on this uploaded data")
    if has_labels:
        rows = []
        for name, m in models.items():
            p = m.predict(X_scaled)
            pr = m.predict_proba(X_scaled)[:, 1]
            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_true, p),
                "AUC": roc_auc_score(y_true, pr),
                "Precision": precision_score(y_true, p, zero_division=0),
                "Recall": recall_score(y_true, p, zero_division=0),
                "F1": f1_score(y_true, p, zero_division=0),
                "MCC": matthews_corrcoef(y_true, p),
            })
        comp_df = pd.DataFrame(rows).set_index("Model").round(4)
        st.dataframe(comp_df.style.highlight_max(axis=0, color="#1B4A44"), use_container_width=True)
        winner = comp_df["MCC"].idxmax()
        st.markdown(f'<span class="risk-pill risk-low">Best MCC: {winner}</span>', unsafe_allow_html=True)
    else:
        st.warning("Upload a CSV with the `cardio` label column to compare all models.")

st.markdown("---")
st.caption("BITS Pilani WILP — M.Tech AIML/DSE — Machine Learning Assignment 2")
