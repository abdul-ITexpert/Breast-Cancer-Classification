import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# Load Model and Scaler
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    font-size:40px;
    color:#0E4C92;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.result-good{
    background:#d4edda;
    padding:20px;
    border-radius:12px;
    border-left:8px solid green;
    font-size:22px;
    font-weight:bold;
}

.result-bad{
    background:#f8d7da;
    padding:20px;
    border-radius:12px;
    border-left:8px solid red;
    font-size:22px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🩺 Breast Cancer Classifier")

st.sidebar.markdown("---")

st.sidebar.write("### Model Information")

st.sidebar.success("Algorithm: Logistic Regression")

st.sidebar.success("Feature Scaling: StandardScaler")

st.sidebar.success("Cross Validation Accuracy: 94%")

st.sidebar.markdown("---")

st.sidebar.write("### Project Workflow")

st.sidebar.write("""
✔ Outlier Removal (IQR & Z-Score)

✔ Label Encoding

✔ Feature Selection

✔ StandardScaler

✔ Logistic Regression

✔ Confusion Matrix Heatmap
""")

# ----------------------------
# Main Heading
# ----------------------------
st.markdown("<h1 class='title'>Breast Cancer Prediction System</h1>", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Enter patient measurements to predict whether the tumor is Benign or Malignant.</p>", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# Input Section
# ===========================

st.header("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", value=14.0)
    texture_mean = st.number_input("Texture Mean", value=19.0)
    perimeter_mean = st.number_input("Perimeter Mean", value=91.0)
    area_mean = st.number_input("Area Mean", value=650.0)
    smoothness_mean = st.number_input("Smoothness Mean", value=0.09)
    compactness_mean = st.number_input("Compactness Mean", value=0.10)
    concavity_mean = st.number_input("Concavity Mean", value=0.08)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.05)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.18)
    radius_se = st.number_input("Radius SE", value=0.40)
    perimeter_se = st.number_input("Perimeter SE", value=2.80)
    area_se = st.number_input("Area SE", value=40.0)
    compactness_se = st.number_input("Compactness SE", value=0.02)

with col2:
    concavity_se = st.number_input("Concavity SE", value=0.03)
    concave_points_se = st.number_input("Concave Points SE", value=0.01)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.003)
    radius_worst = st.number_input("Radius Worst", value=16.0)
    texture_worst = st.number_input("Texture Worst", value=25.0)
    perimeter_worst = st.number_input("Perimeter Worst", value=107.0)
    area_worst = st.number_input("Area Worst", value=880.0)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.13)
    compactness_worst = st.number_input("Compactness Worst", value=0.25)
    concavity_worst = st.number_input("Concavity Worst", value=0.27)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.11)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.29)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.08)

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button("🔍 Predict Cancer Type", use_container_width=True)

st.markdown("---")

# ===========================
# Prediction Section
# ===========================

if predict_button:

    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        radius_se,
        perimeter_se,
        area_se,
        compactness_se,
        concavity_se,
        concave_points_se,
        fractal_dimension_se,
        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst
    ]])

    # Apply StandardScaler
    scaled_input = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Prediction Probabilities
    probability = model.predict_proba(scaled_input)

    benign_prob = probability[0][0] * 100
    malignant_prob = probability[0][1] * 100

    st.subheader("📊 Prediction Result")

    if prediction == 0:

        st.markdown(
            """
            <div class='result-good'>
            🟢 Prediction: BENIGN
            <br><br>
            The model predicts that the tumor is likely non-cancerous.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class='result-bad'>
            🔴 Prediction: MALIGNANT
            <br><br>
            The model predicts that the tumor is likely cancerous.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("📈 Prediction Confidence")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Benign Probability",
            value=f"{benign_prob:.2f}%"
        )

    with col2:
        st.metric(
            label="Malignant Probability",
            value=f"{malignant_prob:.2f}%"
        )

    st.progress(float(max(benign_prob, malignant_prob) / 100))

    st.info(
        f"Model Confidence: {max(benign_prob, malignant_prob):.2f}%"
    )

# ===========================
# Footer
# ===========================

st.markdown("---")

st.markdown(
    """
    <div class='footer'>
    🩺 Breast Cancer Classification System<br>
    Built with Streamlit, Logistic Regression & StandardScaler<br>
    Cross Validation Accuracy: 94%
    </div>
    """,
    unsafe_allow_html=True
)