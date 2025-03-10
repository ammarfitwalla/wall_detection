import time
import cv2
import streamlit as st
import uuid
from PIL import Image
import custom_wall_detection
import wall_detection_roboflow_multiple_models_v1 as robolow_model_detections
import numpy as np
import os
import config

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

image_source = None
uploaded_file_path = None

# Determine the source image (uploaded or sample)
if uploaded_file is not None:
    unique_id = str(uuid.uuid4())
    file_name = f"{unique_id}_{uploaded_file.name}"
    with open(os.path.join("user_input_images", file_name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    uploaded_file_path = os.path.join("user_input_images", file_name)  # set the path
    image_source = uploaded_file_path  
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
            processed_sub_images = [robolow_model_detections.process_image(config.API_URL, config.API_KEY, image_source, sub_image, config.roboflow_models) for sub_image in sub_images]

            # Stitch the processed sub-images back together
            stitched_image = custom_wall_detection.stitch_images(
                processed_sub_images, rows, cols, sub_height, sub_width
            )
        else:
            # Process the whole image if no splitting is necessary
            stitched_image, object_counts = robolow_model_detections.process_image(config.API_URL, config.API_KEY, image_source, image, config.roboflow_models)

        time.sleep(0.5)
        # Convert the processed NumPy array to an image

        stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(stitched_image)

        # Display the processed image
        st.write("**Processed Image with Detected Walls:**")
        st.image(processed_image, caption="Processed Image")

        # Add explanatory boxes
        # st.markdown(
        #     """
        #     <style>
        #     .description-box {
        #         display: flex;
        #         align-items: center;
        #         margin-bottom: 10px;
        #     }
        #     .color-box {
        #         width: 20px;
        #         height: 20px;
        #         margin-right: 10px;
        #     }
        #     .red-box { background-color: red; }
        #     .green-box { background-color: green; }
        #     .blue-box { background-color: blue; }
        #     .purple-box { background-color: purple; }
        #     .orange-box { background-color: orange; }
        #     </style>
        #     <div class="description-box">
        #         <div class="color-box orange-box"></div>
        #         <p>Orange boxes represent detected detected walls.</p>
        #     </div>
        #     <div class="description-box">
        #         <div class="color-box red-box"></div>
        #         <p>Red boxes represents detected doors.</p>
        #     </div>
        #     <div class="description-box">
        #         <div class="color-box blue-box"></div>
        #         <p>Blue boxes represent detected bathrooms.</p>
        #     </div>
        #     <div class="description-box">
        #         <div class="color-box green-box"></div>
        #         <p>Green lines represent detected sinks.</p>
        #     </div>
        #     <div class="description-box">
        #         <div class="color-box purple-box"></div>
        #         <p>Purple boxes represent detected toilets.</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )
        # st.markdown(
        #                 f"""
        #                 <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: black;">
        #                     <h4>Detected Object Counts:</h4>
        #                     <ul>
        #                     {"".join([f"<li><strong>{key.capitalize()}:</strong> {value}</li>" for key, value in object_counts.items()])}
        #                     </ul>
        #                 </div>
        #                 """,
        #                 unsafe_allow_html=True,
        #             )

        if not object_counts:
            st.markdown("<h3 style='color: white;'>No Objects Detected!</h3>", unsafe_allow_html=True)
        else:        
            color_map = {
                'walls': '🟧',
                'doors': '🔴',
                'bathroom': '🔵',
                'sink': '🟢',
                'toilet': '🟣'
            }

            st.write("**Detected Objects:**")

            # Dynamically generate markdown based on object_counts
            for obj, count in object_counts.items():
                color_emoji = color_map.get(obj, '⬜')  # Default to white if not found
                st.markdown(f"{color_emoji} **{obj.capitalize()}:** {count}")
    except Exception as e:
        st.markdown("<h3 style='color: white;'>No Objects Detected!</h3>", unsafe_allow_html=True)
        st.error(f"Error: {str(e)}")
