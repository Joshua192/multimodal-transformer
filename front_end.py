import streamlit as st
from PIL import Image
import io
import torch
import PIL
import cv2
from mess.py import 


model_ = 
model.load_state_dict(torch.load("./models/saved_models/no_train_model", weights_only=False))
# Your model's processing function
def process_image(image: Image.Image):
    # convert image to PIL
    # image = Image.open(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(image)
    return model(image)
    # return "Processed: Image size is {}x{}".format(image.width, image.height)

st.title("Image Processing App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width =True)

    # Process the image
    result = process_image(image)
    
    # Display the result
    st.write("Model Output:")
    st.success(result)
