import shutil
import time
import cv2
import pandas as pd
import streamlit as st
import uuid
from PIL import Image
import custom_wall_detection
import wall_detection_roboflow_multiple_models_v1 as robolow_model_detections
import numpy as np
import os
import config

# Title of the app
st.title("Floor Plan Object Detection")

# Define the path to sample images
sample_images_path = "sample_images"
sample_images = [f for f in os.listdir(sample_images_path) if f.endswith(("jpg", "jpeg", "png"))]

# Dropdown menu to select sample images
st.sidebar.write("## Try Sample Images")
selected_sample = st.sidebar.selectbox("", ["None"] + sample_images)

# File uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

image_source = None
uploaded_file_path = None

# Determine the source image (uploaded or sample)
# Ensure the base directory exists
base_dir = "user_input_images"
os.makedirs(base_dir, exist_ok=True)

unique_id = str(uuid.uuid4())  # Generate a unique ID for every session
user_folder = os.path.join(base_dir, unique_id)  # Create user-specific folder

if "uploaded_file" in locals() and uploaded_file is not None:
    os.makedirs(user_folder, exist_ok=True)  # Ensure the folder exists
    file_name = uploaded_file.name  # Keep the original filename
    file_path = os.path.join(user_folder, file_name)  # Define the path

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image_source = file_path  # Set image source
    st.image(uploaded_file, caption="Uploaded Image")

elif selected_sample != "None":
    os.makedirs(user_folder, exist_ok=True)  # Ensure the folder exists
    sample_image_path = os.path.join(sample_images_path, selected_sample)
    file_name = selected_sample  # Keep the sample filename
    file_path = os.path.join(user_folder, file_name)  # Define the path

    shutil.copy(sample_image_path, file_path)  # Copy sample image to the new folder

    image_source = file_path  # Set image source
    st.image(image_source, caption=f"Sample Image: {selected_sample}")

else:
    st.info("Upload an image -or- select a sample image from the left sidebar to proceed.")
    st.stop()

with st.spinner("Processing the image..."):
    try:
        # Read the selected or uploaded image as a NumPy array
        image = np.array(Image.open(image_source).convert("L"))

        # Determine how many parts to split the image into based on its size
        rows, cols = custom_wall_detection.determine_split(image)

        time.sleep(0.5)
        # Split the image into sub-images if necessary
        # print(f"Rows: {rows}, Cols: {cols}")
        if rows > 1 or cols > 1:
            sub_images, sub_height, sub_width = custom_wall_detection.split_image(image, rows, cols)

            processed_sub_images, object_counts = [], {}
            for i, sub_image in enumerate(sub_images):
                image_path = os.path.join(user_folder, f"sub_image_{str(i)}.png")
                cv2.imwrite(image_path, sub_image)
                processed_sub_images_object_count_list = robolow_model_detections.process_image(config.API_URL, config.API_KEY, image_path, sub_image, config.roboflow_models)
                processed_sub_images.append(processed_sub_images_object_count_list[0])

                # Merge object counts correctly
                for obj, count in processed_sub_images_object_count_list[1].items():
                    object_counts[obj] = object_counts.get(obj, 0) + count

                # object_counts.append(processed_sub_images_object_count_list[1])

            # print(f"Object Counts: {object_counts}")

                # processed_sub_images_object_count_list.append(robolow_model_detections.process_image(config.API_URL, config.API_KEY, image_path, sub_image, config.roboflow_models))

            # # Process each sub-image individually
            # processed_sub_images_object_count_list = [robolow_model_detections.process_image(config.API_URL, config.API_KEY, image_source, sub_image, config.roboflow_models) for sub_image in sub_images]
            # print(f"Processed Sub-Images: {len(processed_sub_images_object_count_list)}")

            # processed_sub_images = [image[0] for image in processed_sub_images_object_count_list if image is not None]
            # object_counts = [obj[1] for obj in processed_sub_images_object_count_list if obj is not None]

            # for processed_sub_image, object_counts in processed_sub_images:
            #     print(type(processed_sub_image))

            #     # processed_image = cv2.cvtColor(processed_sub_image, cv2.COLOR_BGR2RGB)
            #     # processed_sub_image = Image.fromarray(processed_sub_image)
            #     cv2.imshow("Image", processed_sub_image)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()


            # print("Stitching the sub-images back together...")

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
        # st.write("**Processed Image with Detected Walls:**")
        st.markdown("<hr style='border: 2px solid white; margin: 20px 0;'>", unsafe_allow_html=True)

        st.markdown("<h2 style='text-align: center; color: white;'>Processed Image with All Detected Objects</h2>", unsafe_allow_html=True)
        
        st.image(processed_image, caption="Processed Image")

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

            # st.write("**Detected Objects:**")
            st.markdown("<hr style='border: 2px solid white; margin: 20px 0;'>", unsafe_allow_html=True)

            st.markdown("<h2 style='text-align: center; color: white;'>All Detected Objects</h2>", unsafe_allow_html=True)

            # Dynamically generate markdown based on object_counts
            for obj, count in object_counts.items():
                color_emoji = color_map.get(obj, '⬜')  # Default to white if not found
                st.markdown(f"{color_emoji} **{obj.capitalize()}:** {count}")

            # CSV File Download Button
            csv_path = os.path.join(user_folder, "output.csv")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)  # Read CSV file
                csv_data = df.to_csv(index=False).encode('utf-8')  # Convert to CSV format

                # st.markdown("<hr style='border: 2px solid white; margin: 30px 20px;'>", unsafe_allow_html=True)
                # st.markdown("<h3 style='text-align: center; color: white;'>📥 Download Data in .csv Format</h3>", unsafe_allow_html=True)
                
                st.download_button(
                    label="📂 Download CSV",
                    data=csv_data,
                    file_name="otuput.csv",
                    mime="text/csv"
                )

            st.markdown("<hr style='border: 2px solid white; margin: 30px 0;'>", unsafe_allow_html=True)
            

            # Display detected object images below the main output image
            # st.write("**Detected Object Breakdown:**")
            st.markdown("<h2 style='text-align: center; color: white;'>🔍 Detected Objects Breakdown</h2>", unsafe_allow_html=True)

            # Iterate through detected objects and display corresponding images
            for obj, count in object_counts.items():
                st.markdown("<hr style='border: 1px solid white; margin: 30px 0;'>", unsafe_allow_html=True)
                obj_folder = os.path.join(user_folder, obj)  # Folder containing images for this object
                all_files = os.listdir(obj_folder)  # List all files in the folder
                color_emoji = color_map.get(obj, '⬜')  # Default to white if not found
                st.markdown(f"{color_emoji} **{obj.capitalize()}:** {count}")
                for file in all_files:
                    if file.endswith(".png"):
                        obj_image_path = os.path.join(obj_folder, file)  # Expected image file
                        if os.path.exists(obj_image_path):
                            st.image(obj_image_path) #, caption=f"{obj.capitalize()} (Count: {count})", use_column_width=True)
                        else:
                            st.write(f"⚠️ No image found for {obj.capitalize()}")
                
    except Exception as e:
        st.markdown("<h3 style='color: white;'>No Objects Detected!</h3>", unsafe_allow_html=True)
        st.error(f"Error: {str(e)}")
