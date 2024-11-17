import streamlit as st
from PIL import Image
import custom_wall_detection
import numpy as np

# Title of the app
st.title("Wall Detection Image Uploader")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image")

    with st.spinner("Processing the image..."):
        try:
            # Read the uploaded image as a NumPy array
            image = np.array(Image.open(uploaded_file).convert("L"))

            # Determine how many parts to split the image into based on its size
            rows, cols = custom_wall_detection.determine_split(image)

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

            # Convert the processed NumPy array to an image
            processed_image = Image.fromarray(stitched_image)

            # Display the processed image without saving it to disk
            st.write("**Processed Image with Detected Walls:**")
            st.image(processed_image, caption="Processed Image")
        except Exception as e:
            st.write("**No Walls detected!**")
            st.error(f"Error: {str(e)}")
