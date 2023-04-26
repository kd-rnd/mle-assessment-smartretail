import requests
import streamlit as st
from PIL import Image
import io
import base64
import json
import time

# Define the API endpoint URL
API_URL = "http://localhost:8000/process_image"

# Define the Streamlit app
st.title("Pose Estimation Demo")

# Add an input to upload an image
uploaded_files  = st.file_uploader("Upload one or more image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# If an image is uploaded, process it
if uploaded_files :
    output = []
    image_files=[]
    for image in uploaded_files:
        # Open the uploaded image with Pillow
        bytes_data = image.read()
        image_files.append(("images", bytes_data))
        # # Display the uploaded image
        # st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Submit"):
        
        with st.spinner("Generating..."):
            # Send a POST request to the API endpoint with the image data
            response = requests.post(API_URL, files=image_files)
        if response.status_code == 200:
            processed_images = response.json()['processed_images']
            json_pred_str = response.json()['pred_result']
            # json_pred = json.loads(json_pred_str)

            # Loop through the processed images and display each one
            for i, processed_image_bytes in enumerate(processed_images):
                # Decode the base64 data into bytes
                processed_image_bytes = base64.b64decode(processed_image_bytes)

                processed_pil_image = Image.open(io.BytesIO(processed_image_bytes))

                # Display the processed image
                st.image(processed_pil_image, caption=f"Processed Image {i+1}", use_column_width=True)
        
            st.download_button(
                label="Download JSON",
                file_name="result.json",
                mime="application/json",
                data=json_pred_str,
                help='Download the prediction result in json format'
            )
