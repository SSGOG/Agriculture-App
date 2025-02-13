import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Machine Learning for Maize Turcicum Leaf Blight",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load model from file
def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Load ensemble models & scaler
ensemble_model_path = "ensemble_model.pkl"
if os.path.exists(ensemble_model_path):
    with open(ensemble_model_path, "rb") as f:
        scaler, xgb_model, rf_model, bagging_model = pickle.load(f)
else:
    st.error("⚠️ Error: Ensemble model file not found!")

# Background Image
background_image_url = "https://raw.githubusercontent.com/SSGOG/Agriculture-App/main/maahaha1.JPG"
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("{background_image_url}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title & Introduction
st.title("🌾 Machine Learning for Forecasting Maize Turcicum Leaf Blight")
st.markdown("""
### Introduction
Turcicum Leaf Blight (TLB) is a critical foliar disease that significantly impacts maize production.
This app uses machine learning models to forecast the disease index (PDI) based on weather parameters.
""")

# Sidebar Input Parameters
st.sidebar.header("Input Parameters")
with st.sidebar.expander("Adjust Parameters", expanded=True):
    temp_max = st.sidebar.number_input("Temperature Max (°C):", min_value=0.0, step=0.1)
    temp_min = st.sidebar.number_input("Temperature Min (°C):", min_value=0.0, step=0.1)
    rh_max = st.sidebar.number_input("Relative Humidity Max (%):", min_value=0.0, step=0.1)
    rh_min = st.sidebar.number_input("Relative Humidity Min (%):", min_value=0.0, step=0.1)
    wind_speed = st.sidebar.number_input("Wind Speed (km/h):", min_value=0.0, step=0.1)
    sun_shine = st.sidebar.number_input("Sun Shine (hrs):", min_value=0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm):", min_value=0.0, step=0.1)

# Model Selection
models = {
    "Random Forest Regressor": "random_forest_model.pkl",
    "XGBoost Regressor": "xgb_model.pkl",
    "Decision Trees Regressor": "DTR.pkl",
    "K-Nearest Neighbours Regressor": "knn_model.pkl",
    "Bagging Regressor": "BR_model.pkl",
    "Extra Trees Regressor": "ETR_linear_regression_model.pkl",
    "Support Vector Regression": "svr_model.pkl",
}
model_choice = st.sidebar.selectbox("🔍 Select a Machine Learning Model:", list(models.keys()))

# Prepare input features
input_features = np.array([[temp_max, temp_min, rh_max, rh_min, wind_speed, sun_shine, rainfall]])

# Check if all inputs are zero
def all_inputs_zero(inputs):
    return np.all(inputs == 0)

# Predict with Selected Model
if st.sidebar.button("Predict"):
    if all_inputs_zero(input_features):
        st.warning("⚠️ Please enter valid input values before predicting.")
    else:
        try:
            model_path = models.get(model_choice)
            if model_path and os.path.exists(model_path):
                model = load_pickle(model_path)
                input_features_scaled = scaler.transform(input_features) if "scaler" in locals() else input_features
                prediction = model.predict(input_features_scaled)[0]
                st.success(f"Predicted Disease Index (PDI): {prediction:.2f}")
            else:
                st.error(f"⚠️ Error: Model file for {model_choice} not found!")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Predict with Ensemble Model
if st.sidebar.button("Predict with Ensemble Model"):
    if all_inputs_zero(input_features):
        st.warning("⚠️ Please enter valid input values before predicting.")
    else:
        try:
            input_features_scaled = scaler.transform(input_features) if "scaler" in locals() else input_features
            pred_xgb = xgb_model.predict(input_features_scaled)
            pred_rf = rf_model.predict(input_features_scaled)
            pred_bagging = bagging_model.predict(input_features_scaled)
            ensemble_prediction = (pred_xgb + pred_rf + pred_bagging) / 3
            st.success(f"Predicted Disease Index (PDI) (Ensemble): {ensemble_prediction[0]:.2f}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
### Developed for Farmers 🌱
This platform enables farmers to take early and informed action, reducing crop losses and optimizing resource use. 
Integrating ML into an accessible interface supports global food security efforts. 🚜
""")
