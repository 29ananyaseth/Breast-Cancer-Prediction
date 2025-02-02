import streamlit as st
import pandas as pd
import joblib

# Set up the Streamlit app (must be the first command in the script)
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

# Inject custom CSS to change background color to blue and set the font to Manrope
st.write(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope&display=swap');
    
    body {
        background-color: #87CEEB;  /* Light Blue color */
        font-family: 'Manrope', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model and scaler
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set up the Streamlit app
st.title("Breast Cancer Tumor Classification")
st.subheader("Predict whether the tumor is Benign or Malignant based on the input features.")

# Add a brief description for the app
st.markdown("""
This app uses machine learning (SVM) to predict whether a breast tumor is benign or malignant based on features like:
- **Radius Mean**
- **Perimeter Mean**
- **Area Mean**
- **Smoothness Mean**
- **Concavity Mean**
- **Symmetry Mean**

### How to Use:
1. Enter values for the features below.
2. Click **Predict** to see the result.
""")

# Create interactive input fields for the features
st.markdown("### Input Features (Enter values):")

# Input fields for features
radius_mean = st.number_input('Radius Mean', min_value=0.0, max_value=30.0, value=15.4, step=0.1)
perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, max_value=200.0, value=85.0, step=0.1)
area_mean = st.number_input('Area Mean', min_value=0.0, max_value=5000.0, value=530.0, step=1.0)
smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, max_value=0.2, value=0.097, step=0.001)
concavity_mean = st.number_input('Concavity Mean', min_value=0.0, max_value=0.5, value=0.057, step=0.001)
symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, max_value=0.5, value=0.182, step=0.001)

# Button to make prediction
if st.button('Predict', key='predict_button'):
    # Prepare the input data as a DataFrame
    new_data = pd.DataFrame({
        'radius_mean': [radius_mean],
        'perimeter_mean': [perimeter_mean],
        'area_mean': [area_mean],
        'smoothness_mean': [smoothness_mean],
        'concavity_mean': [concavity_mean],
        'symmetry_mean': [symmetry_mean]
    })
    
    # Standardize the new data
    new_data_scaled = scaler.transform(new_data)

    # Make prediction using the loaded SVM model
    prediction = svm_model.predict(new_data_scaled)

    # Display the result with a more visually appealing message
    if prediction == 1:
        result = "Malignant"
        st.markdown(f'<p style="font-size: 20px; color: red; font-weight: bold;">The tumor is predicted to be <span style="color: red;">**{result}**</span>. Please consult a healthcare professional.</p>', unsafe_allow_html=True)
    else:
        result = "Benign"
        st.markdown(f'<p style="font-size: 20px; color: green; font-weight: bold;">The tumor is predicted to be <span style="color: green;">**{result}**</span>. It\'s likely non-cancerous, but please consult a healthcare professional.</p>', unsafe_allow_html=True)
    
    # Add some extra information
    st.markdown("### Key Features Used for Prediction")
    st.write(f"**Radius Mean:** {radius_mean}")
    st.write(f"**Perimeter Mean:** {perimeter_mean}")
    st.write(f"**Area Mean:** {area_mean}")
    st.write(f"**Smoothness Mean:** {smoothness_mean}")
    st.write(f"**Concavity Mean:** {concavity_mean}")
    st.write(f"**Symmetry Mean:** {symmetry_mean}")
    
    # Add some guidance or recommendations after prediction
    st.markdown("""
    **Note:**
    - This is a machine learning prediction and should not be used as a substitute for medical advice.
    - If you have concerns about breast cancer, please consult with a healthcare professional.
    """)

# Footer section with a friendly touch
st.markdown("---")
st.markdown("""
Made with ❤️ by Ananya | Streamlit App
""")
