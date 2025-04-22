import streamlit as st
import pickle as pkl
import numpy as np
import os
from PIL import Image
try:
    model = pkl.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: model.pkl file not found. Please run train_model.py first.")
    st.stop()

st.title("Breast Cancer Prediction")
img=Image.open(os.path.join('static', 'image_1.jpeg'))
st.image(img, caption="Breast Cancer")
st.write("This app predicts whether a breast cancer tumor is malignant or benign.")
st.write("Please enter the features of the tumor below:")
st.write("Note: Features should be separated by commas.")

st.write("Example: 1.0, 2.0, 3.0, 4.0, 5.0")
st.write("Note: The model expects 30 features. Please provide all of them.")

features = st.text_input("Enter features:")
if st.button("Predict"):
    if features:
        try:
            features = [float(x) for x in features.split(',')]
            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features)
            prediction_proba = prediction_proba[0][1] * 100
            if prediction[0] == 1:
                result = f"Prediction: Malignant \n (Probability: {prediction_proba:.2f}%)"
            else:
                result = f"Prediction: Benign \n (Probability: {100-prediction_proba:.2f}%)"
            st.success(result)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter the features.")
# Note: The model.pkl file should be in the same directory as this script.
# If not, please provide the correct path to the model file.


