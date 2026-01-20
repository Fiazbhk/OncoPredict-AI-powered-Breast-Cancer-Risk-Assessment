import streamlit as st
import joblib
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="OncoPredict",
    layout="centered"
)

# -----------------------------
# LOAD MODEL ASSETS
# -----------------------------
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# FEATURE ORDER (CANONICAL)
# -----------------------------
FEATURES = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error","smoothness error",
    "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
    "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

# -----------------------------
# SIDEBAR (Paperaxis-style)
# -----------------------------
st.sidebar.title("❉ OncoPredict")

st.sidebar.markdown(
    """
**AI-powered Breast Cancer Risk Assessment**

This system uses a machine learning model trained on clinical diagnostic
features to predict whether a tumor is **benign or malignant**.

**Tech Stack:** Python, ML, SVM, Streamlit
"""
)

st.sidebar.markdown("### Capabilities")
st.sidebar.markdown(
    """
- Manual clinical feature input  
- CSV-based patient record analysis  
- Probability-based diagnosis  
- Research-grade inference pipeline  
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
[GitHub](https://github.com/Fiazbhk) | 
[LinkedIn](https://www.linkedin.com/in/fiazbhk/) |
[LeetCode](https://leetcode.com/u/muhammadfiazbhk/)
"""
)

# -----------------------------
# TITLE
# -----------------------------
st.title("❉ OncoPredict")

# -----------------------------
# TOP HORIZONTAL MENU
# -----------------------------
selected_tab = option_menu(
    menu_title=None,
    options=["Predict Cancer Type", "About the Model"],
    icons=["activity", "info-circle"],
    orientation="horizontal",
    default_index=0
)

# -----------------------------
# TAB 1: PREDICTION
# -----------------------------
if selected_tab == "Predict Cancer Type":
    st.subheader("Clinical Feature Input")

    # CSV Upload (Paperaxis-like flow)
    uploaded_file = st.file_uploader(
        "Upload Patient Record (CSV)",
        type=["csv"]
    )

    prefill = {}

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower()

        rename_map = {
            "radius_mean": "mean radius",
            "texture_mean": "mean texture",
            "perimeter_mean": "mean perimeter",
            "area_mean": "mean area",
            "smoothness_mean": "mean smoothness",
            "compactness_mean": "mean compactness",
            "concavity_mean": "mean concavity",
            "concave points_mean": "mean concave points",
            "symmetry_mean": "mean symmetry",
            "fractal_dimension_mean": "mean fractal dimension",

            "radius_se": "radius error",
            "texture_se": "texture error",
            "perimeter_se": "perimeter error",
            "area_se": "area error",
            "smoothness_se": "smoothness error",
            "compactness_se": "compactness error",
            "concavity_se": "concavity error",
            "concave points_se": "concave points error",
            "symmetry_se": "symmetry error",
            "fractal_dimension_se": "fractal dimension error",

            "radius_worst": "worst radius",
            "texture_worst": "worst texture",
            "perimeter_worst": "worst perimeter",
            "area_worst": "worst area",
            "smoothness_worst": "worst smoothness",
            "compactness_worst": "worst compactness",
            "concavity_worst": "worst concavity",
            "concave points_worst": "worst concave points",
            "symmetry_worst": "worst symmetry",
            "fractal_dimension_worst": "worst fractal dimension"
        }

        df.rename(columns=rename_map, inplace=True)

        if all(f in df.columns for f in FEATURES):
            prefill = df.iloc[0].to_dict()
            st.success("CSV loaded and mapped successfully")
        else:
            st.error("CSV does not contain required features")

    # Manual Input (3-column layout)
    inputs = {}
    cols = st.columns(3)

    sections = [
        ("Mean Features", FEATURES[0:10], 0),
        ("Error Features", FEATURES[10:20], 1),
        ("Worst Features", FEATURES[20:30], 2)
    ]

    for title, feats, col_idx in sections:
        with cols[col_idx]:
            st.subheader(title)
            for f in feats:
                label = f.replace("mean ", "").replace("error", "err").replace("worst ", "").title()
                inputs[f] = st.number_input(
                    label,
                    value=float(prefill.get(f, 0.0)),
                    format="%.4f",
                    key=f"num_{f}"
                )

    st.divider()

    if st.button("Generate Diagnostic Result"):
        X = np.array([inputs[f] for f in FEATURES]).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        if pred == 1:
            st.error(f"### ASSESSMENT: MALIGNANT\nProbability: {prob:.4f}")
        else:
            st.success(f"### ASSESSMENT: BENIGN\nProbability: {1 - prob:.4f}")

# -----------------------------
# TAB 2: ABOUT
# -----------------------------
if selected_tab == "About the Model":
    st.subheader("Model Information")

    st.markdown(
        """
**Algorithm:** Support Vector Machine (RBF Kernel)  
**Preprocessing:** StandardScaler  
**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset  
**Target:**  
- 0 → Benign  
- 1 → Malignant  

**Notes:**
- Predictions are probabilistic, not diagnostic
- Intended for educational and research use
- Not a substitute for medical evaluation
"""
    )

