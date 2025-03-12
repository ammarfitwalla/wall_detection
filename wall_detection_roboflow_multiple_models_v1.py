import os
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import cv2
import numpy as np
import pandas as pd

# Function to initialize the client
def initialize_client(api_url, api_key):
    return InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key
    )

# Function to run inference for any model
def run_inference(client, image_path, model_id, confidence_threshold):
    config = InferenceConfiguration(confidence_threshold=confidence_threshold)
    client.configure(config)
    result = client.infer(image_path, model_id=model_id)
    return result

import os
import cv2
import numpy as np

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, raw_input_image, process_image_path, result, label_colors, object_counts, color_mapping):
    input_image = raw_input_image.copy()  # Copy the original image
    object_images = {}  # Dictionary to store images for each object type
    bounding_boxes = []  # List to store bounding box details for CSV

    alpha_main = 0.15  # Lighter transparency for main image
    alpha_segregated = 0.3  # Slightly darker transparency for segregated images

    for prediction in result['predictions']:
        object_name = prediction['class'].lower()

        # Skip objects not in label colors
        if object_name not in label_colors:
            continue

        color = label_colors[object_name]

        # Create a new subfolder for this object type if it doesn't exist
        object_folder = os.path.join(process_image_path, object_name)
        os.makedirs(object_folder, exist_ok=True)

        # Extract coordinates
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = x1 + int(prediction['width'])
        y2 = y1 + int(prediction['height'])

        # Store bounding box details for CSV
        bounding_boxes.append({
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
            "object_detected": object_name
        })

        # Initialize a separate image for each object type
        if object_name not in object_images:
            object_images[object_name] = input_image.copy()

        # Create overlays for both main and segregated images
        overlay_main = image.copy()
        overlay_object = object_images[object_name].copy()

        # Fill bounding boxes with solid color
        cv2.rectangle(overlay_main, (x1, y1), (x2, y2), color, -1)  # Main image
        cv2.rectangle(overlay_object, (x1, y1), (x2, y2), color, -1)  # Segregated image

        # Blend overlays with transparency
        cv2.addWeighted(overlay_main, alpha_main, image, 1 - alpha_main, 0, image)  # Main image (lighter)
        cv2.addWeighted(overlay_object, alpha_segregated, object_images[object_name], 1 - alpha_segregated, 0, object_images[object_name])  # Segregated images (stronger)

        # Draw bounding box outlines (to maintain visibility)
        cv2.rectangle(object_images[object_name], (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Update object count
        object_counts[object_name] = object_counts.get(object_name, 0) + 1
        color_mapping[object_name] = color

    # Save all object-specific images after looping
    for obj_name, obj_img in object_images.items():
        save_path = os.path.join(process_image_path, obj_name, f"{obj_name}.png")
        cv2.imwrite(save_path, obj_img)

    # Convert bounding box list to DataFrame and save as CSV
    df = pd.DataFrame(bounding_boxes)
    csv_path = os.path.join(process_image_path, "output.csv")
    df.to_csv(csv_path, index=False)

    # return csv_path



# Function to run the pipeline
def process_image(api_url, api_key, image_path, image, models):
    client = initialize_client(api_url, api_key)
    
    # Load the image
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Initialize variables for object counts and color mapping
    object_counts = {}
    color_mapping = {}

    process_image_path = image_path.split(os.sep)[:-1]
    process_image_path = os.sep.join(process_image_path)

    input_image = image.copy()

    # Run inference on all models and draw bounding boxes
    for model_name, config in models.items():
        if not config['display']:
            continue
        result = run_inference(client, image_path, config['model_id'], config['confidence_threshold'])
        draw_bounding_boxes(image, input_image, process_image_path, result, config['label_colors'], object_counts, color_mapping)

    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print("Inference completed")

    # # Extend the image to add a white space for text
    # height, width, _ = image.shape
    # new_width = width + 300  # Add 300 pixels for object count space
    # extended_image = np.ones((height, new_width, 3), dtype=np.uint8) * 255
    # extended_image[:, :width] = image  # Copy original image to the left side

    # # Draw object counts in the white space
    # y_offset = 50
    # cv2.putText(extended_image, "Detected Objects:", (width + 10, 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # # Display counts with the same color as bounding boxes
    # for obj, count in object_counts.items():
    #     color = color_mapping[obj]
    #     cv2.putText(extended_image, f"{obj}: {count}",
    #                 (width + 10, y_offset),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    #     y_offset += 25

    return image, object_counts

    # return extended_image  # Return processed image for further use

# Model configurations
# models = {
#     "walls": {
#         "model_id": "walldetector2/1",
#         "confidence_threshold": 0.1,
#         "label_colors": {"walls": (204, 100, 102)},  # Blue
#         "comments": "8/10 result for walls",
#         "display": True,
#     },
#     "walls2": {
#         "model_id": "floortest2/1",
#         "confidence_threshold": 0.01,
#         "label_colors": {"wall": (255, 0, 0)},  # Blue
#         "comments": "7/10 result for walls",
#         "display": False,
#     },
#     "doors": {
#         "model_id": "doors-vetjc/1",
#         "confidence_threshold": 0.1,
#         "label_colors": {"doors": (0, 0, 255)},  # Red
#         "comments": "8/10 result for doors",
#         "display": True,
#     },
#     # "objects": {
#     #     "model_id": "yolo-obb-1/1",
#     #     "confidence_threshold": 0.1,
#     #     "label_colors": {"object": (0, 255, 0)},  # Green
#     #     "comments": "7/10 result for doors and walls",
#     #     "display": False,
#     # },
#     "toilet": {
#         "model_id": "bathroom-tyaoe/9",
#         "confidence_threshold": 0.2,
#         "label_colors": {
#             "sink": (0, 255, 0),  # Green
#             "bathroom": (255, 165, 0),  # Orange
#             # "door": (0, 0, 255),  # Red
#             "toilet": (128, 0, 128)  # Purple
#         },
#         "comments": "7/10 result for toilets",
#         "display": True,
#     },
# }

# Example usage
# api_key = "aCfWpAkGRSWAsm9Suz2u"
# input_image = "sample_images/test_room_cubicasa.jpeg"
# output_image = process_image(api_key, input_image, models)

# Now you can pass `output_image` to another function instead of displaying or saving it.
