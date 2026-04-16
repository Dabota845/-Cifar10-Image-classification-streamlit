import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
  page_title="CIFAR-10 AI Classifier",
  page_icon="💻",
  layout="centered"
)

#CIFAR-10 class labels
class_names=['airplane', 
             'automobile', 
             'bird', 
             'cat', 
             'deer', 
             'dog', 
             'frog', 
             'horse', 
             'ship', 
             'truck'
            ]

#Load model
def load_model():
    model = tf.keras.models.load_model("regularized_cnn_cifar10.keras")
    return model

model = load_model()

st.title("CIFAR-10 Image Classification App")
st.write("Upload an image and the trained CNN model will classify it.")

#upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  st.image(image, caption="Uploaded Image", use_column_width=True)
  image = image.resize((32,32))
  img_array = np.array(image) / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  with st.spinner("Analyzing image..."):
    prediction = model.predict(img_array, verbose=0)
  predicted_index = np.argmax(prediction)
  predicted_class = class_names[predicted_index]
  confidence = np.max(prediction)
  st.success(f"Predicted Class: **{predicted_class}**")
  st.progress(float(confidence))
  st.write(f"Confidence: **{confidence*100:.2f}%**")
  






