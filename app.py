import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load your trained model saved with joblib
model_path = Path(__file__).parent / "25RP18587.joblib"
if not model_path.is_file():
    st.error(f"Model file not found: {model_path}")
    st.stop()

loaded_model = joblib.load(model_path)

# App title
st.title("🌾 Crop Yield Prediction (Temperature‑only)")

# Input for temperature
temperature = st.number_input(
    "Temperature (°C):",
    min_value=-10.0,
    max_value=50.0,
    step=0.1,
    value=25.0
)

# Prepare input dataframe
input_df = pd.DataFrame([{"Temperature": temperature}])

# Prediction when button pressed
if st.button("Predict Yield"):
    try:
        prediction = loaded_model.predict(input_df)
        st.success(f"Predicted Crop Yield: {prediction[0]:.2f} tons per hectare")
    except Exception as e:
        st.error(f"Prediction error: {e}")
