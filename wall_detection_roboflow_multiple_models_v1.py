from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import cv2

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

# Function to draw bounding boxes
def draw_bounding_boxes(image, result, label_colors, object_counts, color_mapping):
    for prediction in result['predictions']:
        object_name = prediction['class'].lower()

        # Skip objects not in label colors
        if object_name not in label_colors:
            continue
        color = label_colors[object_name]

        # Extract coordinates
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = x1 + int(prediction['width'])
        y2 = y1 + int(prediction['height'])

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Count the objects
        object_counts[object_name] = object_counts.get(object_name, 0) + 1
        color_mapping[object_name] = color

# Function to run the pipeline
def process_image(api_url, api_key, image_path, image, models):
    client = initialize_client(api_url, api_key)
    
    # Load the image
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    object_counts = {}
    color_mapping = {}

    # Run inference on all models and draw bounding boxes
    for model_name, config in models.items():
        if not config['display']:
            continue
        result = run_inference(client, image_path, config['model_id'], config['confidence_threshold'])
        draw_bounding_boxes(image, result, config['label_colors'], object_counts, color_mapping)

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
