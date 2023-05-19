# Import necessary libraries
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gdown

# Set the desired height and width of the images
img_height = 224
img_width = 224

# Define class names
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
               'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
               'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 
               'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
               'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
               'Tomato___healthy']

# URL for the model file in Google Drive
url = 'https://drive.google.com/uc?id=1bTLtAgLvJLFr2WE4gzzln9Ko4h4tP5-v'
output = 'resnet50_model.h5'
gdown.download(url, output, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(output)

# Function to predict class of the image
def predict(image):
    # Resize the image and convert it to a numpy array
    image = image.resize((img_height, img_width))
    img_array = np.array(image)

    # Display the original image before preprocessing
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title('Original Image')
    st.pyplot(fig)  # Show the plot in streamlit

    # Preprocess the image with the same preprocessing function used during training
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # Reshape the image to be fed into the model for prediction
    img_array = tf.expand_dims(img_array, axis=0)

    # Predict the class of the image
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    return predicted_class  # Return the predicted class

# Title of the streamlit app
st.title('"DeepLeaf" - Plant Leaf Classifier')

# Provide an interface for users to upload an image from their device or enter an image URL
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open the image file

    # Display the uploaded image
    st.image(image, caption='Uploaded Leaf Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict the class of the image
    label = predict(image)

    # Display the predicted class
    st.write(f'Predicted class: {label}')
