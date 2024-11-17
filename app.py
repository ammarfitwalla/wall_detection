import time
import streamlit as st
from PIL import Image
import custom_wall_detection
import numpy as np
import os

# Title of the app
st.title("Wall Detection Image Uploader")

# Define the path to sample images
sample_images_path = "sample_images"
sample_images = [f for f in os.listdir(sample_images_path) if f.endswith(("jpg", "jpeg", "png"))]

# Dropdown menu to select sample images
st.sidebar.write("## Try Sample Images")
selected_sample = st.sidebar.selectbox("Choose a sample image", ["None"] + sample_images)

# File uploader
uploaded_file = st.file_uploader("Upload your own image", type=["jpg", "jpeg", "png"])

# Determine the source image (uploaded or sample)
if uploaded_file is not None:
    image_source = uploaded_file
    st.image(uploaded_file, caption="Uploaded Image")
elif selected_sample != "None":
    image_source = os.path.join(sample_images_path, selected_sample)
    st.image(image_source, caption=f"Sample Image: {selected_sample}")
else:
    st.info("Please upload an image or select a sample image from the left sidebar to proceed.")
    st.stop()

with st.spinner("Processing the image..."):
    try:
        # Read the selected or uploaded image as a NumPy array
        image = np.array(Image.open(image_source).convert("L"))

        # Determine how many parts to split the image into based on its size
        rows, cols = custom_wall_detection.determine_split(image)

        time.sleep(0.5)
        # Split the image into sub-images if necessary
        if rows > 1 or cols > 1:
            sub_images, sub_height, sub_width = custom_wall_detection.split_image(image, rows, cols)

            # Process each sub-image individually
            processed_sub_images = [custom_wall_detection.process_sub_image(sub_image) for sub_image in sub_images]

            # Stitch the processed sub-images back together
            stitched_image = custom_wall_detection.stitch_images(
                processed_sub_images, rows, cols, sub_height, sub_width
            )
        else:
            # Process the whole image if no splitting is necessary
            stitched_image = custom_wall_detection.process_sub_image(image)

        time.sleep(0.5)
        # Convert the processed NumPy array to an image
        processed_image = Image.fromarray(stitched_image)

        # Display the processed image
        st.write("**Processed Image with Detected Walls:**")
        st.image(processed_image, caption="Processed Image")

        # Add explanatory boxes
        st.markdown(
            """
            <style>
            .description-box {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }
            .color-box {
                width: 20px;
                height: 20px;
                margin-right: 10px;
            }
            .red-box { background-color: red; }
            .green-box { background-color: green; }
            </style>

            <div class="description-box">
                <div class="color-box red-box"></div>
                <p>Red boxes indicate text detected by OCR.</p>
            </div>
            <div class="description-box">
                <div class="color-box green-box"></div>
                <p>Green lines represent detected walls.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.markdown("<h3 style='color: white;'>No Walls Detected!</h3>", unsafe_allow_html=True)
        # st.error(f"Error: {str(e)}")
