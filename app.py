import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil

# Set the desired height and width of the images
img_height = 224
img_width = 224

# List of class names
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
               'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 
               'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
               'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Potato___healthy', 'Raspberry___healthy', 
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot']

# Load the trained model
model = tf.keras.models.load_model('C:/Users/exman/Desktop/Python backup/Capstone/Models/resnet50_model.h5') # Path to stored trained model

# Function to predict class of the image
def predict(image):
    # Get the original image dimensions and size
    original_dims = image.size
    st.write(f"Original image dimensions: {original_dims[0]} x {original_dims[1]}")
    st.write(f"Original image size: {uploaded_file.size} bytes")

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

    # Start the timer and get the current memory usage
    start_time = time.time()
    mem_before = psutil.virtual_memory().used

    # Predict the class of the image
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    # Get the elapsed time and used memory
    elapsed_time = time.time() - start_time
    mem_after = psutil.virtual_memory().used
    mem_used = mem_after - mem_before

    # Print the prediction time and used resources
    st.write(f"Time taken for prediction: {elapsed_time} seconds")
    st.write(f"Memory used for prediction: {mem_used} bytes")

    return predicted_class  # Return the predicted class

# Title of the streamlit app
st.title('"DeepLeaf" - Plant Leaf Classifier')

# Note for future update
st.write("""
Please note, at present, our application is able to predict the following plant leaf conditions based on the limitations of our training dataset:

- Healthy plant leaves: Apple, Blueberry, Corn, Grape, Peach, Potato, Raspberry, Strawberry.
- Diseased plant leaves: Apple, Cherry, Corn, Grape, Orange, Peach, Pepper, Squash, Strawberry, Tomato.

This list represents the scope of our current classification abilities and we aim to expand our dataset for better predictions in the future.
""")

# Provide an interface for users to upload an image from their device or enter an image URL
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open the image file

    # Display the uploaded image
    st.image(image, caption='Uploaded Leaf Image.', use_column_width=False, width=378)
    st.write("")
    st.write("Classifying...")

    # Predict the class of the image
    label = predict(image)

    # Display the predicted class
    st.write(f'It looks like {label}')
