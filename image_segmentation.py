import streamlit as st
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

INPUT_SIZE = (640, 640)  

def predict(image):
    model = YOLO('model/yolov8n.pt') 

    results = model.predict(image)

    result_images = []
    for result in results:
        result_img = result.plot()  
        result_images.append(result_img)

    return result_images

def main():
    st.title("Construction site Image Segmentation")

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:

        if uploaded_file.type not in ['image/jpeg', 'image/png']:
            st.error("Only inputs of jpg, jpeg, and png format are allowed.")
            return

        image = Image.open(uploaded_file)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_resized = image.resize(INPUT_SIZE)

        st.image(image_resized, caption='Resized Image', use_column_width=True)

        if st.button("Predict"):
            result_images = predict(np.array(image_resized))
            for idx, result_img in enumerate(result_images):
                plt.imshow(result_img)
                plt.axis('off')  
                st.pyplot(plt)

        if st.checkbox("Show Original Image"):
            st.image(image, caption='Original Image', use_column_width=True)

    if st.button("Back to Homepage"):
        st.session_state.page = "homepage"

def run():
    main()

if __name__ == "__main__":
    run()
