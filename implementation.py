import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To reduce TensorFlow logging


import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas



# Load the model
try:
    model = load_model('LeNet.h5', compile=False)
except Exception as e:
    st.error(f"Failed to load the model: {e}")
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def main():
    # Define a function to preprocess the drawn image
    def preprocess_image(image):
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image = np.array(image).astype('float32')  # Convert to numpy array and change type to float32
        image = image / 255.0  # Normalize to [0, 1]
        image = image.reshape(1, 28, 28, 1)  # Reshape for model input (batch size, height, width, channels)
        return image

    # Streamlit app layout
    st.title('Handwritten Digit Recognition')
    st.write('Draw a digit in the box below and press "Predict"')

    # Create a blank canvas
    canvas_result = st_canvas(
        fill_color='black',
        width=280,
        height=280,
        stroke_width=30,
        stroke_color='white',
        background_color='black',
        update_streamlit=True,
        key='canvas'
    )

    # Display the canvas image and handle prediction
    if st.button('Predict'):
        if canvas_result.image_data is not None:
            # Convert canvas image data to PIL Image
            image_data = canvas_result.image_data
            pil_image = Image.fromarray(image_data.astype(np.uint8))

            # Preprocess the image
            preprocessed_image = preprocess_image(pil_image)
            
            # Make prediction
            try:
                prediction = model.predict(preprocessed_image)
                predicted_class = np.argmax(prediction, axis=1)[0]
                # Display the prediction
                st.write(f'The model predicts: {predicted_class}')
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.write('Please draw a digit first.')

if __name__ == '__main__':
    main()
