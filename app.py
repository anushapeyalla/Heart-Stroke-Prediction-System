import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load model and files (cached so they don't reload on every interaction)
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("Desicion_heart_pred.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

st.set_page_config(page_title="Stroke Prediction", page_icon="❤️")

st.title("❤️ Stroke Prediction System")
st.write("Enter the patient's details below.")

try:
    model, scaler, columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Could not load model files. Make sure 'Desicion_heart_pred.pkl', "
        "'scaler.pkl', and 'columns.pkl' are in the same folder as this app.\n\n"
        f"Details: {e}"
    )
    st.stop()

# ----------------------------
# User Inputs
# ----------------------------

id = st.number_input("Patient ID", min_value=1, value=1, step=1)

gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0)

hypertension = st.selectbox("Hypertension", [0, 1])

heart_disease = st.selectbox("Heart Disease", [0, 1])

ever_married = st.selectbox("Ever Married", ["Yes", "No"])

work_type = st.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)

Residence_type = st.selectbox(
    "Residence Type",
    ["Urban", "Rural"]
)

avg_glucose_level = st.number_input(
    "Average Glucose Level",
    min_value=0.0,
    value=100.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["formerly smoked", "never smoked", "smokes", "Unknown"]
)

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict"):

    input_data = pd.DataFrame({
        "id": [id],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "gender_Male": [int(gender == "Male")],
        "ever_married_Yes": [int(ever_married == "Yes")],
        "work_type_Never_worked": [int(work_type == "Never_worked")],
        "work_type_Private": [int(work_type == "Private")],
        "work_type_Self-employed": [int(work_type == "Self-employed")],
        "work_type_children": [int(work_type == "children")],
        "Residence_type_Urban": [int(Residence_type == "Urban")],
        "smoking_status_formerly smoked": [int(smoking_status == "formerly smoked")],
        "smoking_status_never smoked": [int(smoking_status == "never smoked")],
        "smoking_status_smokes": [int(smoking_status == "smokes")],
    })

    # Ensure all expected columns exist (fill any missing with 0)
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Arrange columns in the order the scaler was fit on (columns.pkl)
    input_data = input_data[columns]

    try:
        # Scale -- this array has however many columns the scaler expects (e.g. 17)
        input_scaled = scaler.transform(input_data)

        # The model may have been fit on a subset of those columns (e.g. 16,
        # missing "id"). If the model remembers its training column names,
        # use that to select the right subset/order before predicting.
        if hasattr(model, "feature_names_in_"):
            model_feature_names = list(model.feature_names_in_)
            col_indices = [columns.index(c) for c in model_feature_names]
            input_for_model = input_scaled[:, col_indices]
        else:
            # Fallback: model doesn't expose feature names (older sklearn or
            # trained on a plain numpy array). Best guess: drop "id" if the
            # feature-count mismatch is exactly 1 and "id" is present.
            if len(columns) - model.n_features_in_ == 1 and "id" in columns:
                drop_idx = columns.index("id")
                keep_idx = [i for i in range(len(columns)) if i != drop_idx]
                input_for_model = input_scaled[:, keep_idx]
            else:
                raise ValueError(
                    f"Can't automatically align features: scaler produced "
                    f"{input_scaled.shape[1]} columns but model expects "
                    f"{model.n_features_in_}, and the model has no "
                    f"feature_names_in_ to resolve which to drop."
                )

        # Predict
        prediction = model.predict(input_for_model)[0]

        # Get probability of the "stroke" class (label 1) safely,
        # regardless of the order model.classes_ happens to be in
        proba_row = model.predict_proba(input_for_model)[0]
        if 1 in model.classes_:
            stroke_idx = list(model.classes_).index(1)
        else:
            stroke_idx = int(np.argmax(proba_row))
        probability = proba_row[stroke_idx]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Risk of Stroke")
        else:
            st.success("✅ Low Risk of Stroke")

        st.write(f"**Probability of Stroke:** {probability:.2%}")

    except Exception as e:
        st.error(f"Something went wrong while making the prediction: {e}")