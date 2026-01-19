import streamlit as st
import pickle
import numpy as np
import os

# ------------------------------
# 1. LOAD MODEL
# ------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')
try:
    linear_model = pickle.load(open(MODEL_PATH, 'rb'))
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}")

# ------------------------------
# 2. STREAMLIT APP UI
# ------------------------------
st.title("Linear Regression Prediction App")

st.write("Enter feature values separated by commas (e.g., 5.1, 3.5, 1.4, 0.2)")

input_text = st.text_input("Features", "")

if st.button("Predict"):
    try:
        # Convert input to numpy array
        features = np.array([float(x) for x in input_text.split(",")]).reshape(1, -1)
        prediction = linear_model.predict(features)[0]
        st.success(f"Predicted value: {prediction:.4f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

