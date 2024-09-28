import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

model = keras.models.load_model('model/cats_vs_dogs_model.h5') 

class_names = ['Cat', 'Dog']  

def predict(image):
    image_resized = image.resize((224, 224)) 
    img_array = np.array(image_resized).astype('float32')  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    predictions = model.predict(img_array)
    label_index = np.argmax(predictions, axis=1)[0]  
    return class_names[label_index]

def run():  
    st.title("Cats vs Dogs Image Classification")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            label = predict(image)
            st.write(f"The model predicts: **{label}**")

    if st.button("Back to Homepage"):
        st.session_state.page = "homepage"

if __name__ == "__main__":
    run()  
