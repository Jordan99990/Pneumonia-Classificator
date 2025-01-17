import os
import io
import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATHS = {
    "cnn_custom": "models/best_model.keras",
    "cnn_vgg16": "models/best_model_pretrained.keras",
}

models = {name: load_model(path) for name, path in MODEL_PATHS.items()}

uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_image(model, image_array):
    prediction = model.predict(image_array)
    label = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
    percentage = prediction[0][0] * 100
    color = 'red' if label == 'Pneumonia' else 'green'
    return label, percentage, color

if uploaded_img is not None:
    image = Image.open(io.BytesIO(uploaded_img.read())).convert('RGB')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    image_array = preprocess_image(image)
    
    predictions = {name: predict_image(model, image_array) for name, model in models.items()}
    
    col1, col2, col3, col4 = st.columns([1, 0.2, 2, 1])
    with col3:
        for name, (label, percentage, color) in predictions.items():
            st.markdown(f'{name.replace("_", " ").title()} Prediction: <span style="color:{color}">{label}</span> ({percentage:.2f}%)', unsafe_allow_html=True)