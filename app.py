import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Credit Approval Prediction",
    page_icon="💳",
    layout="centered"
)

# =====================================
# LOAD TRAINED MODEL
# =====================================

MODEL_PATH = Path("models/creditworthiness_model.pkl")

st.title("Credit Approval Prediction")
st.write(
    "Enter the 15 application features to obtain a prediction "
    "from the trained machine learning model."
)

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run python train_model.py first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

# =====================================
# INPUT FORM
# =====================================

st.subheader("Applicant Information")

st.info(
    "This demonstration uses the UCI Credit Approval dataset. "
    "Its original features are anonymized, so the inputs below "
    "are generic feature values, not verified financial amounts."
)

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        a1 = st.number_input("Feature 1", value=30.0)
        a2 = st.number_input("Feature 2", value=1.0)
        a3 = st.number_input("Feature 3", value=5.0)
        a4 = st.selectbox("Feature 4", ["u", "y", "l", "t"])
        a5 = st.selectbox("Feature 5", ["g", "p", "gg"])

        a6 = st.number_input("Feature 6", value=1.0)
        a7 = st.selectbox("Feature 7", ["c", "d", "cc", "i", "j", "k", "m", "r", "q", "w", "x", "e", "aa", "ff"])

        a8 = st.number_input("Feature 8", value=1.0)
        a9 = st.selectbox("Feature 9", [0, 1])
        a10 = st.number_input("Feature 10", value=0.0)

    with col2:
        a11 = st.selectbox("Feature 11", [0, 1])
        a12 = st.selectbox("Feature 12", ["g", "p", "s"])
        a13 = st.selectbox("Feature 13", ["g", "p", "s"])
        a14 = st.number_input("Feature 14", value=100.0)
        a15 = st.number_input("Feature 15", value=1.0)

    submitted = st.form_submit_button("Predict Credit Approval")

# =====================================
# MAKE PREDICTION
# =====================================

if submitted:

    input_data = pd.DataFrame(
        [[
            a1, a2, a3, a4, a5,
            a6, a7, a8, a9, a10,
            a11, a12, a13, a14, a15
        ]],
        columns=list(range(15))
    )

    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Predicted class: Approved (+)")
        else:
            st.warning("Predicted class: Not approved (-)")

        st.write(
            f"Estimated probability of the approved class: "
            f"{probabilities[1] * 100:.2f}%"
        )

        st.caption(
            "This is a demonstration of a machine learning classifier, "
            "not a real lending decision or a reliable assessment of "
            "an individual's creditworthiness."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")