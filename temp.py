# Import necessary packages
import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

# Load the saved model
model = keras.models.load_model(r"C:\Users\91733\Desktop\GitHub\leaf-identification\leaf-identification\BC.h6")

# Define a function to check if an image is a leaf
def is_leaf(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define lower and upper bounds for green color in HSV
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])

    # Threshold the image to extract green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Calculate the proportion of green pixels in the image
    green_pixels = np.count_nonzero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    proportion_green = green_pixels / total_pixels

    # If the proportion of green pixels is above a certain threshold, consider it a leaf
    if proportion_green > 0.1:
        return True
    else:
        return False

# Define the function to make predictions
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255
    prediction = model.predict(x)
    return prediction

# Create the Streamlit app
def app():
    st.title('Leaf Identification')

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Make a prediction on the uploaded image and display the predicted class label
    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            image_path = uploaded_file.name
            with open(image_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Load the image and check if it is a leaf
            image = cv2.imread(image_path)
            if is_leaf(image):
                prediction = predict(image_path)
                class_labels = ['Acer Palmatum','Cedrus Deodara','Cercis Chinensis','Citrus Reticulata Blanco','Ginkgo Biloba','Liriodendron Chinense','Nerium Oleander']
                predicted_class = class_labels[np.argmax(prediction)]
                st.write(f'This is a {predicted_class}')
            else:
                st.write('This is not a leaf.')
        else:
            st.write('Please upload an image file.')

if __name__ == '__main__':
    app()
