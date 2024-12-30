import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
from PIL import Image
import numpy as np
import pandas as pd
import os

# Prepare data
model_path = "models/20241230-095627-full-dataset.h5"
labels_df = pd.read_csv('labels.csv')
labels_array = np.array(labels_df['breed'])
unique_breeds = np.unique(labels_array)

# Load the model
model = keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})

# Helper functions
def process_image(image, IMG_SIZE=224):
    # Convert image to tensor
    image = tf.image.decode_image(image, channels=3)

    # Set the image shape
    image.set_shape([None, None, 3])
    
    # Normalize the image
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize the image
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    
    return image

def create_data_batches(X, BATCH_SIZE=1):
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch

def get_pred_label(prediction_probs):
    return unique_breeds[np.argmax(prediction_probs)]

# Set the page config
st.set_page_config(layout="wide", page_title="Dog Breed Predictions", page_icon="🐕")
hide_default_format = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_default_format, unsafe_allow_html=True)


# Create the StreamLit interface
st.markdown("<h1 style='text-align: center;'>🐕 Dog Breed Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload an image to get started!</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read file contents
    image_bytes = uploaded_file.read()

    # Display the uploaded image
    col1, col2, col3 = st.columns([0.4, 0.2, 0.4])
    with col2:
        st.image(image_bytes, width=300)

    # Process the image
    image_batch = create_data_batches([image_bytes])

    # Make prediction
    prediction = model.predict(image_batch)

    # Get the predicted label
    pred_label = get_pred_label(prediction)

    # Get the confidence
    confidence = np.max(prediction)

    # Display the predicted label
    st.markdown(f"<h2 style='text-align: center;'>Prediction: {pred_label}</h2>", unsafe_allow_html=True)
    # Display the progress bar centered
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.progress(float(confidence))
    st.markdown("</div>", unsafe_allow_html=True)

    # Center the probability display using HTML in Markdown
    st.markdown(f"<h3 style='text-align: center;'>Probability: {confidence * 100:.2f}%</h3>", unsafe_allow_html=True)

