import streamlit as st
from PIL import Image
import io

# Your model's processing function
def process_image(image: Image.Image):
    # Dummy model output
    return "Processed: Image size is {}x{}".format(image.width, image.height)

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
