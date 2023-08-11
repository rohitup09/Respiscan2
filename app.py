import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet as rt
from PIL import Image
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

# Load the saved model
model = load_model('ModelWeights.h5')

# Define class labels
class_labels = ['ACA', 'N', 'SCC']  # Replace with your actual class labels

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input size
    imagen = img_to_array(image)  # Normalize pixel values to [0, 1]
    imageb = np.expand_dims(imagen, axis=0)  # Add batch dimension
    return imageb.copy()

def main():
    st.title("Image Classification App")
    st.write("Upload an image for prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Predict"):
            # Preprocess the image
            input_image = preprocess_image(image)
            processed_image = rt.preprocess_input(input_image)
            # Perform prediction
            predictions = model.predict(processed_image)
            # st.write(predictions)
            predicted_class = np.argmax(predictions, axis = 1)
            #Adenocarcinoma, Normal, Squamous Cell Carcinoma
            if(predicted_class == [0]):
                st.write(f"Predicted class: Adenocarcinoma")
            elif(predicted_class == [1]):
                st.write(f"Predicted class: Benign Tissue")
            else:
                st.write(f"Predicted class: Squamous Cell Carcinoma")

if __name__ == "__main__":
    main()
