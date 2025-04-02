import streamlit as st
import pickle
import numpy as np
import base64
import os

# Set page configuration
st.set_page_config(
    page_title="Machine Learning for Maize Turcicum Leaf Blight",
    layout="wide",
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"
    st.session_state.input_valid = False
    st.session_state.input_features = None

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "TLB Disease Information", "Contact Us"])
st.session_state.page = page

# Function to load model from file
def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Load ensemble models & scaler
try:
    ensemble_model_path = r"ensemble_model.pkl"
    if os.path.exists(ensemble_model_path):
        with open(ensemble_model_path, "rb") as f:
            scaler, xgb_model, rf_model, bagging_model = pickle.load(f)
    else:
        st.sidebar.error("⚠️ Ensemble model file not found!")
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading ensemble model: {str(e)}")

# Function to encode image as Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# Background Image
background_image_url = "https://raw.githubusercontent.com/SSGOG/Agriculture-App/main/Untitled1.JPG"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    .content-box {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Home Page
if st.session_state.page == "Home":
    st.title("🌾 Machine Learning for Forecasting Maize Turcicum Leaf Blight")
    st.markdown("""
        <div class="content-box">
        <style>
        .content-box {
                color: white;
                background-color: rgba(0, 0, 0, 0.5);
        }
        </style>
        <h2>Welcome to the Maize Turcicum Leaf Blight Prediction App!</h2><br>        
        <h3>Introduction</h3>
        <p>Turcicum Leaf Blight (TLB) is a critical foliar disease that significantly impacts maize production.
        This app uses machine learning models to forecast the disease index (PDI) based on weather parameters.</p><br>
        <h3>How to Use the App</h3>
        1. Enter all the weather parameters in the Input Parameters menu.<br>
        2. Click on the "Predict" button to get the predicted disease index.<br>
        3. You can also choose to predict using an ensemble model for better accuracy.<br>
        4. You can also view the symptoms and solutions for TLB by navigating to the "Symptoms & Solutions" page.<br>
        5. If you have any questions or feedback, please reach out to us via the "Contact Us" page.<br>        
        </div>
    """, unsafe_allow_html=True)

    # Input Parameters
    st.markdown("""
        <div class="content-box">
        <h3>Input Parameters</h3>
        <p>Enter the weather parameters below to predict the disease index:</p>
        """, unsafe_allow_html=True)
    
    with st.form("input_form"):
        temp_max = st.number_input("Temperature Max (°C):", min_value=0.0, step=0.1)
        temp_min = st.number_input("Temperature Min (°C):", min_value=0.0, step=0.1)
        rh_max = st.number_input("Relative Humidity Max (%):", min_value=0.0, step=0.1)
        rh_min = st.number_input("Relative Humidity Min (%):", min_value=0.0, step=0.1)
        wind_speed = st.number_input("Wind Speed (km/h):", min_value=0.0, step=0.1)
        sun_shine = st.number_input("Sun Shine (hrs):", min_value=0.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm):", min_value=0.0, step=0.1)
        submitted = st.form_submit_button("Submit")
    
        if submitted:
            if temp_max == 0 or temp_min == 0 or rh_max == 0 or rh_min == 0 or wind_speed == 0 or sun_shine == 0 or rainfall == 0:
                st.error("⚠️ Error: All input parameters must be greater than zero!")
                st.session_state.input_valid = False
            else:
                st.session_state.input_valid = True
                st.session_state.input_features = np.array([[temp_max, temp_min, rh_max, rh_min, wind_speed, sun_shine, rainfall]])
                st.success("Input parameters submitted successfully!")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Model Selection
    models = {
        "Random Forest Regressor": r"random_forest_model.pkl",
        "XGBoost Regressor": r"xgb_model.pkl",
        "Decision Trees Regressor": r"DTR.pkl",
        "K-Nearest Neighbours Regressor": r"knn_model.pkl",
        "Bagging Regressor": r"BR_model.pkl",
        "Extra Trees Regressor": r"ETR_linear_regression_model.pkl",
        "Support Vector Regression": r"svr_model.pkl",
    }
    model_choice = st.selectbox("🔍 Select a Machine Learning Model:", list(models.keys()))
    
    # Prediction buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict"):
            if not st.session_state.input_valid:
                st.error("Please submit valid input parameters first!")
            else:
                try:
                    model_path = models.get(model_choice)
                    if model_path and os.path.exists(model_path):
                        model = load_pickle(model_path)
                        if "scaler" in locals():
                            input_features_scaled = scaler.transform(st.session_state.input_features)
                        else:
                            input_features_scaled = st.session_state.input_features
                        prediction = model.predict(input_features_scaled)[0]
                        st.success(f"Predicted Disease Index (PDI): {prediction:.2f}")
                    else:
                        st.error(f"⚠️ Error: Model file for {model_choice} not found!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        if st.button("Predict with Ensemble Model"):
            if not st.session_state.input_valid:
                st.error("Please submit valid input parameters first!")
            else:
                try:
                    if "scaler" in locals():
                        input_features_scaled = scaler.transform(st.session_state.input_features)
                    else:
                        input_features_scaled = st.session_state.input_features
                    pred_xgb = xgb_model.predict(input_features_scaled)
                    pred_rf = rf_model.predict(input_features_scaled)
                    pred_bagging = bagging_model.predict(input_features_scaled)
                    ensemble_prediction = (pred_xgb + pred_rf + pred_bagging) / 3
                    st.success(f"Predicted Disease Index (PDI) (Ensemble): {ensemble_prediction[0]:.2f}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Symptoms & Solutions Page
elif st.session_state.page == "TLB Disease Information":
    st.title("🌱 TLB Disease Information")
    st.markdown("""
        <div class='content-box'>
        <style>
        .content-box {
                color: white;
                background-color: rgba(0, 0, 0, 0.5);
        }
        </style>
        <h2>Symptoms of Maize Turcicum Leaf Blight (TLB)</h2>
        1. Small, water-soaked, greyish-green lesions appear on lower leaves.<br>
        2. Lesions enlarge into long, elliptical, necrotic streaks with tan centres and reddish-brown margins.<br>
        3. Severe infections cause lesions to coalesce, leading to extensive leaf blight.<br>
        4. Premature drying of leaves reduces photosynthesis, affecting grain filling and yield.

        ---

        ## 🔬 Best Fungicide Strategy Based on Disease Severity

        | Disease Severity | Symptoms | Recommended Fungicide & Dosage | Application Frequency |
        |------------------|----------|-------------------------------|------------------------|
        | **Mild** (1-10%) | Small, scattered lesions on lower leaves | Azoxystrobin 18.2% + Difenoconazole 11.4% SC (1.0 ml/L) | 1 spray at early symptoms |
        | **Moderate** (11-30%) | Expanding lesions affecting multiple leaves | Azoxystrobin 18.2% + Cyproconazole 7.3% SC (1.0 ml/L) | 2 sprays at 10-12 day intervals |
        | **Severe** (>30%) | Large coalescing lesions, premature drying | Pyraclostrobin 133 g/L + Epoxiconazole 50 g/L SE (1.5 ml/L) | 2-3 sprays at 7-10 day intervals |
        | **Epidemic** (>50%) | Extensive leaf damage, rapid spread | Azoxystrobin 18.2% + Difenoconazole 11.4% SC (1.0 ml/L) + Pyraclostrobin 20% WG (1.0 g/L) in rotation | 3 sprays at 7-day intervals |

        ---            

        ## 🧪 Fungicide Recommendations for Turcicum Leaf Blight in Maize

        | Fungicide | Mean PDI (%) | Mean PSOC (%) | Effectiveness | Recommendation |
        |-----------|--------------|---------------|---------------|----------------|
        | Azoxystrobin 18.2% + Difenoconazole 11.4% SC | 9.4 | 89.5 | 🥇 Most Effective | ✅ Highly Recommended |
        | Azoxystrobin 18.2% + Cyproconazole 7.3% SC | 13.9 | 84.6 | 🥈 Very Effective | ✅ Recommended |
        | Pyraclostrobin 133 g/L + Epoxiconazole 50 g/L SE | 26.6 | 70.4 | 🥉 Moderate Control | 🔄 Can be used in rotation |
        | Carbendazim 12% + Mancozeb 63% WP | 35.0 | 61.1 | ⭐⭐ Moderate | Alternate with systemic fungicides |
        | Kresoxim-methyl 4.3% SC | 40.5 | 55.0 | ⭐ Low Efficacy | ❌ Not preferred for severe infections |
        | Zineb 75% WP | 52.7 | 41.3 | ❌ Least Effective | 🚫 Not recommended |
                    
        ---

        ## 🛠️ Best Practices for Effective Disease Management
        1. Apply fungicide sprays early to prevent severe infection.<br>  
        2. Rotate fungicides to avoid resistance build-up.<br>  
        3. Maintain field sanitation to reduce pathogen spread.<br>  
        4. Monitor weather conditions as high humidity and moderate temperatures favour TLB.<br>  
        5. Use recommended dosage and proper spraying methods to ensure maximum efficacy.
        ---
        </div>
    """, unsafe_allow_html=True)

# Contact Us Page
elif st.session_state.page == "Contact Us":
    st.title("📩 Contact Us")
    st.write("For inquiries, please reach out via email or phone.")
    st.markdown("""
        <div class='content-box'>
        <style>
        .content-box {
                color: white;
                background-color: rgba(0, 0, 0, 0.5);
        }
        </style>        
        📲 Contact Details

        **Dr. Jadesha G., PhD**  
        Plant Pathologist   
        University of Agricultural Sciences, Bengaluru, Karnataka, India.  
        📧 Email: jadesha.uasb@gmail.com  

        **Dr. Deepak D., PhD**  
        Professor  
        Department of Mechatronics  
        Manipal Institute of Technology  
        Manipal Academy of Higher Education, Manipal-576104, India.  
        📧 Email: deepak.d@manipal.edu  

        **Dr. Anupkumar Bongale**  
        Associate Professor  
        Department of Artificial Intelligence and Machine Learning  
        Symbiosis Institute of Technology, Pune-412115, India  
        📧 Email: anupkumar.bongale@sitpune.edu.in  

        **Shreyans Gadekar**   
        Department of Artificial Intelligence and Machine Learning  
        Symbiosis Institute of Technology, Pune-412115, India  
        📧 Email: shreyans.gadekar.btech2023@sitpune.edu.in

        </div>
    """, unsafe_allow_html=True)
