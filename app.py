import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

# Set the height and width of the images
img_height = 224
img_width = 224

# Read the training data directory and extract class names
train_dir = 'C:/Users/exman/Desktop/Python backup/Capstone/train'
class_names = sorted(os.listdir(train_dir))

# Load the model
model = tf.keras.models.load_model('C:/Users/exman/Desktop/Python backup/Capstone/Models/resnet50_model.h5')

def predict(image):
    # Resize the image and convert it to a numpy array
    image = image.resize((img_height, img_width))
    img_array = np.array(image)

    # Plot the original image before preprocessing
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title('Original Image')
    st.pyplot(fig)

    # Preprocess the image using the same preprocessing function used during training
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # Reshape the image
    img_array = tf.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    return predicted_class

st.title('"DeepLeaf" - Plant Leaf Classifier')

# Provide a user interface for users to either upload an image from their device or enter an image URL
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Leaf Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict the class of the image
    label = predict(image)

    st.write(f'Predicted class: {label}')
