import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# Load your trained model
model = load_model('LeNet.h5')

# Define a function to preprocess the drawn image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize to [0, 1]
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

# Streamlit app layout
st.title('Handwritten Digit Recognition')
st.write('Draw a digit in the box below and press "Predict"')

# Create a blank canvas
canvas_result = st_canvas(
    fill_color='white',
    width=280,
    height=280,
    stroke_width=10,
    stroke_color='black',
    background_color='white',
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
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        # Display the prediction
        st.write(f'The model predicts: {predicted_class}')
    else:
        st.write('Please draw a digit first.')
