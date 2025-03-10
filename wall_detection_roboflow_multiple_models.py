# from inference_sdk import InferenceConfiguration, InferenceHTTPClient
# import cv2
# import numpy as np

# # Input image
# # input_image = "sample_images/test_wall.png"
# # input_image = "sample_images/test_room_cubicasa.jpeg"
# input_image = "sample_images/test_wall_3.png"

# # Function to initialize the client
# def initialize_client(api_key):
#     return InferenceHTTPClient(
#         api_url="https://detect.roboflow.com",
#         api_key=api_key
#     )

# # Function to run inference for any model
# def run_inference(client, model_id, confidence_threshold):
#     config = InferenceConfiguration(confidence_threshold=confidence_threshold)
#     client.configure(config)
#     result = client.infer(input_image, model_id=model_id)
#     return result

# # ✅ Function to draw bounding boxes and capture color for text
# def draw_bounding_boxes(image, result, color, label_name, object_counts, color_mapping):
#     for prediction in result['predictions']:
#         # Extract coordinates
#         x = int(prediction['x'] - prediction['width'] / 2)
#         y = int(prediction['y'] - prediction['height'] / 2)
#         w = int(prediction['width'])
#         h = int(prediction['height'])

#         # Draw the rectangle with custom color
#         cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

#         # Count the objects and store their color
#         object_name = prediction['class']
#         object_counts[object_name] = object_counts.get(object_name, 0) + 1

#         # Store the color for the text count later
#         if object_name not in color_mapping:
#             color_mapping[object_name] = color

# # ✅ Step 1: Initialize client
# api_key = "aCfWpAkGRSWAsm9Suz2u"
# client = initialize_client(api_key)

# # ✅ Step 2: Your EXACT models dictionary
# models = {
#     "walls": {
#         "model_id": "walldetector2/1",
#         "confidence_threshold": 0.05,
#         "color": (255, 0, 0),  # Blue color for walls
#         "label": "Wall",
#         "comments": "8/10 result for walls",
#         "display": False,
#     },
#     "walls2": {
#         "model_id": "floortest2/1",
#         "confidence_threshold": 0.01,
#         "color": (255, 0, 0),  # Blue color for walls
#         "label": "Wall",
#         "comments": "7/10 result for walls",
#         "display": False,
#     },
#     "doors": {
#         "model_id": "doors-vetjc/1",
#         "confidence_threshold": 0.1,
#         "color": (0, 0, 255),  # Red color for doors
#         "label": "Door",
#         "comments": "8/10 result for doors",
#         "display": False,
#     },
#     # "doors2": {
#     #     "model_id": "door-object-detection/1",
#     #     "confidence_threshold": 0.3,
#     #     "color": (0, 0, 255),  # Red color for doors
#     #     "label": "Door",
#     #     "comments": "8/10 result for doors",
#     #     "display": False,
#     # },
#     "objects": {
#         "model_id": "yolo-obb-1/1",
#         "confidence_threshold": 0.1,
#         "color": (0, 255, 0),  # Green color for objects
#         "label": "Object",
#         "comments": "7/10 result for doors and walls",
#         "display": False,
#     },
#     "toilet": {
#         "model_id": "bathroom-tyaoe/9",
#         "confidence_threshold": 0.3,
#         "color": (0, 255, 0),  # Green color for toilets
#         "label": "toilet",
#         "comments": "7/10 result for toilets",
#         "display": True,
#     },
# }

# # ✅ Step 3: Load the image
# image = cv2.imread(input_image)
# object_counts = {}
# color_mapping = {}

# # ✅ Step 4: Run all models and draw bounding boxes
# for model_name, config in models.items():
#     if not config['display']:
#         continue
#     result = run_inference(client, config['model_id'], confidence_threshold=config['confidence_threshold'])
#     draw_bounding_boxes(image, result, config['color'], config['label'], object_counts, color_mapping)

# # ✅ Step 5: EXTEND THE IMAGE TO THE RIGHT SIDE (Add White Space)
# height, width, _ = image.shape
# new_width = width + 300  # Adding 300 pixels for text space

# # Create a new white image
# extended_image = np.ones((height, new_width, 3), dtype=np.uint8) * 255
# extended_image[:, :width] = image  # Copy the original image to the left side

# # ✅ Step 6: Draw object counts in the white space with their respective colors
# y_offset = 50
# cv2.putText(
#     extended_image, "Detected Objects:", 
#     (width + 10, 20),
#     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
# )

# # ✅ Step 7: Display counts in white space with the SAME color as bounding box
# for obj, count in object_counts.items():
#     color = color_mapping[obj]

#     # Draw the text in the same color as the bounding box
#     cv2.putText(
#         extended_image, f"{obj}: {count}", 
#         (width + 10, y_offset),
#         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
#     )
#     y_offset += 25

# # ✅ Step 8: Draw a vertical black separator line
# cv2.line(
#     extended_image,
#     (width + 5, 0), (width + 5, height),
#     (0, 0, 0), 2
# )

# # ✅ Step 9: Display the image
# cv2.imshow("Output", extended_image)
# cv2.imwrite("output/final_output.png", extended_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import cv2
import numpy as np

# input_image = "sample_images/test_wall.png"
input_image = "sample_images/test_room_cubicasa.jpeg"
# input_image = "sample_images/test_wall_3.png"

# Function to initialize the client
def initialize_client(api_key):
    return InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

# Function to run inference for any model
def run_inference(client, model_id, confidence_threshold):
    config = InferenceConfiguration(confidence_threshold=confidence_threshold)
    client.configure(config)
    result = client.infer(input_image, model_id=model_id)
    return result

# ✅ Function to draw bounding boxes and capture color for text
def draw_bounding_boxes(image, result, label_colors, object_counts, color_mapping):
    """
    Draws bounding boxes only for specified labels if filter_labels is provided.
    Otherwise, draws all detected objects.
    """
    for prediction in result['predictions']:
        object_name = prediction['class'].lower()

        # If filter_labels is provided, ignore objects not in the allowed list
        # if filter_labels and object_name not in filter_labels:
        #     continue

        # Extract coordinates
        x = int(prediction['x'] - prediction['width'] / 2)
        y = int(prediction['y'] - prediction['height'] / 2)
        w = int(prediction['width'])
        h = int(prediction['height'])

        # Get the color for this label
        if object_name not in label_colors:
            continue
        color = label_colors[object_name]
        # color = label_colors.get(object_name, (0, 0, 0))  # Default to black if not defined

        # Draw the rectangle with the label's color
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Count the objects and store their color
        object_counts[object_name] = object_counts.get(object_name, 0) + 1

        # Store the color for the text count later
        if object_name not in color_mapping:
            color_mapping[object_name] = color

# ✅ Step 1: Initialize client
api_key = "aCfWpAkGRSWAsm9Suz2u" 
client = initialize_client(api_key)

# ✅ Step 2: Your EXACT models dictionary with multiple labels per model
models = {
    "walls": {
        "model_id": "walldetector2/1",
        "confidence_threshold": 0.1,
        "label_colors": {"walls": (204, 100, 102)},  # Blue
        "comments": "8/10 result for walls",
        "display": True,
    },
    "walls2": {
        "model_id": "floortest2/1",
        "confidence_threshold": 0.01,
        "label_colors": {"wall": (255, 0, 0)},  # Blue
        "comments": "7/10 result for walls",
        "display": False,
    },
    "doors": {
        "model_id": "doors-vetjc/1",
        "confidence_threshold": 0.1,
        "label_colors": {"doors": (0, 0, 255)},  # Red
        "comments": "8/10 result for doors",
        "display": True,
    },
    # "objects": {
    #     "model_id": "yolo-obb-1/1",
    #     "confidence_threshold": 0.1,
    #     "label_colors": {"object": (0, 255, 0)},  # Green
    #     "comments": "7/10 result for doors and walls",
    #     "display": False,
    # },
    "toilet": {
        "model_id": "bathroom-tyaoe/9",
        "confidence_threshold": 0.2,
        "label_colors": {
            "sink": (0, 255, 0),  # Green
            "bathroom": (255, 165, 0),  # Orange
            # "door": (0, 0, 255),  # Red
            "toilet": (128, 0, 128)  # Purple
        },
        "comments": "7/10 result for toilets",
        "display": True,
        # "filter_labels": ["sink", "bathroom", "door", "toilet"],  # Only detect these objects
    },
}

# ✅ Step 3: Load the image
image = cv2.imread(input_image)
object_counts = {}
color_mapping = {}

# ✅ Step 4: Run all models and draw bounding boxes
for model_name, config in models.items():
    if not config['display']:
        continue
    result = run_inference(client, config['model_id'], confidence_threshold=config['confidence_threshold'])
    draw_bounding_boxes(
        image, result, 
        config['label_colors'], object_counts, color_mapping,
        # filter_labels=config.get("filter_labels")  # Apply filtering only if specified
    )

# ✅ Step 5: EXTEND THE IMAGE TO THE RIGHT SIDE (Add White Space)
height, width, _ = image.shape
new_width = width + 300  # Adding 300 pixels for text space

# Create a new white image
extended_image = np.ones((height, new_width, 3), dtype=np.uint8) * 255
extended_image[:, :width] = image  # Copy the original image to the left side

# ✅ Step 6: Draw object counts in the white space with their respective colors
y_offset = 50
cv2.putText(
    extended_image, "Detected Objects:", 
    (width + 10, 20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
)

# ✅ Step 7: Display counts in white space with the SAME color as bounding box
for obj, count in object_counts.items():
    color = color_mapping[obj]

    # Draw the text in the same color as the bounding box
    cv2.putText(
        extended_image, f"{obj}: {count}", 
        (width + 10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )
    y_offset += 25

# ✅ Step 8: Draw a vertical black separator line
cv2.line(
    extended_image,
    (width + 5, 0), (width + 5, height),
    (0, 0, 0), 2
)

# ✅ Step 9: Display and save the image
cv2.imshow("Output", extended_image)
cv2.imwrite("sample_output_images/roboflow/final_output.png", extended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
