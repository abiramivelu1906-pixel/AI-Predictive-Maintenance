
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
import matplotlib.pyplot as plt

# Load trained models
rf_model = joblib.load('random_forest_compressed.pkl')
xgb_model = joblib.load('xgboost.pkl')
lstm_model = tf.keras.models.load_model('lstm_model.keras')

scaler = joblib.load('scaler.pkl')

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# Dashboard title
st.title("AI-Powered Predictive Maintenance System")

st.write("Aircraft Engine Remaining Useful Life Prediction")

st.divider()

# Model selection
model_name = st.selectbox(
    "Select Prediction Model",
    ["Random Forest", "XGBoost", "LSTM"]
)

# Random Forest and XGBoost
if model_name in ["Random Forest", "XGBoost"]:

    st.subheader("Engine Sensor Readings")

    input_values = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_columns):

        with col1 if i % 2 == 0 else col2:

            input_values[feature] = st.number_input(
                feature,
                value=0.0
            )

    if st.button("Predict RUL"):

        input_df = pd.DataFrame([input_values])

        if model_name == "Random Forest":
            prediction = rf_model.predict(input_df)[0]
        else:
            prediction = xgb_model.predict(input_df)[0]

        st.success("Prediction Completed!")

        st.metric(
            "Predicted Remaining Useful Life",
            f"{prediction:.2f} Cycles"
        )

        if prediction <= 20:
            st.error("Engine Status: Critical")

        elif prediction <= 50:
            st.warning("Engine Status: Warning")

        else:
            st.success("Engine Status: Healthy")

# LSTM prediction
else:

    st.subheader("Upload Engine Sensor History")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file:

        data = pd.read_csv(uploaded_file)

        if set(feature_columns).issubset(data.columns):

            if len(data) >= 30:

                if st.button("Predict RUL"):

                    input_data = data[feature_columns].tail(30)

                    scaled_data = scaler.transform(input_data)

                    sequence = np.array([scaled_data])

                    prediction = lstm_model.predict(
                        sequence,
                        verbose=0
                    )[0][0]

                    st.success("Prediction Completed!")

                    st.metric(
                        "Predicted Remaining Useful Life",
                        f"{prediction:.2f} Cycles"
                    )

            else:
                st.warning("Please upload at least 30 rows.")

        else:
            st.error("Required columns are missing.")

st.divider()

st.caption("NASA C-MAPSS Dataset | Predictive Maintenance Project")
