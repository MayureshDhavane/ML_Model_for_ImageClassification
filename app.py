import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Helper function: Preprocess image for a given model
def preprocess_image(image, target_size, preprocess_func=None):
    """
    Preprocesses the input image for classification models.
    Args:
        image (PIL.Image): Input image.
        target_size (tuple): Desired image size (width, height).
        preprocess_func (function): Optional preprocessing function for the model.
    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    if preprocess_func:
        image_array = preprocess_func(image_array)
    else:
        image_array = image_array.astype('float32') / 255.0  # Normalize
    
    return image_array

# Function: MobileNetV2 ImageNet Classification
def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")
    uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "png"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Classifying...")
            
            with st.spinner("Loading MobileNetV2 model..."):
                model = tf.keras.applications.MobileNetV2(weights='imagenet')
            
            # Preprocess the image
            preprocessed_image = preprocess_image(
                image, target_size=(224, 224),
                preprocess_func=tf.keras.applications.mobilenet_v2.preprocess_input
            )
            
            # Make predictions
            predictions = model.predict(preprocessed_image)
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
            
            st.write("Predictions:")
            for _, label, score in decoded_predictions[0]:
                st.write(f"- **{label.capitalize()}**: {score * 100:.2f}%")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Function: CIFAR-10 Custom Model Classification
def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "png"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Classifying...")
            
            with st.spinner("Loading CIFAR-10 model..."):
                model = tf.keras.models.load_model('model111.h5')
            
            # CIFAR-10 class names
            class_names = [
                'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
            ]
            
            # Preprocess the image
            preprocessed_image = preprocess_image(image, target_size=(32, 32))
            
            # Make predictions
            predictions = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)
            
            st.write(f"Predicted Class: **{class_names[predicted_class]}**")
            st.write(f"Confidence: **{confidence * 100:.2f}%**")
        
        except FileNotFoundError:
            st.error("Model file 'model111.h5' not found. Please ensure it is in the correct location.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Main function for navigation
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ["MobileNetV2 (ImageNet)", "CIFAR-10"])
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

if __name__ == "__main__":
    main()

