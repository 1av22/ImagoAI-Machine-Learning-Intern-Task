import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- Load the saved model, scaler, PCA, and feature names ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the app directory
parent_dir = os.path.dirname(script_dir)

# Construct the path to the 'models' directory
models_dir = os.path.join(parent_dir, 'models')

# Load the model
model = joblib.load(os.path.join(models_dir, 'RF_Model.joblib'))

# Load the scaler
scaler = joblib.load(os.path.join(models_dir, 'Scaler.joblib'))

# Load PCA
pca = joblib.load(os.path.join(models_dir, 'PCA.joblib'))

# Load feature names
feature_names = joblib.load(os.path.join(models_dir, 'feature_names.joblib'))

# --- Streamlit App ---

st.title("Mycotoxin (DON) Prediction in Corn")

st.write("Enter the spectral reflectance values as comma-separated numbers (e.g., `0.1, 0.2, ..., 0.9`)")

input_string = st.text_input("Input Spectral Data")

if st.button("Predict"):
    if input_string:
        try:
            # Split the input string into values
            values = input_string.split(',')

            feature_values = []
            for value in values:
                cleaned_value = value.strip().replace(',', '.')
                feature_values.append(float(cleaned_value))

            # Prepare the input data
            input_data = np.array(feature_values).reshape(1, -1)

            # Create a DataFrame with the *correct* column names
            # Use loaded feature names!
            input_df = pd.DataFrame(input_data, columns=feature_names)

            # Preprocess the data
            input_scaled = scaler.transform(input_df)
            input_pca = pca.transform(input_scaled)

            # Make prediction
            prediction = model.predict(input_pca)[0]

            # Display the prediction
            st.subheader("Predicted DON Concentration:")
            st.write(f"Prediction: {prediction:.4f}")

        except ValueError as e:
            st.error(
                f"Invalid input. Please ensure all values are numeric. Error details: {e}")
        except IndexError:
            st.error("Invalid input. Please provide all feature values.")
    else:
        st.warning("Please enter spectral data.")
