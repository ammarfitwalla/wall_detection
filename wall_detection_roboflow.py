# from inference_sdk import InferenceConfiguration, InferenceHTTPClient
# input_image = "sample_images/test_wall.png"

# # Define your custom confidence threshold (0.2 = 20%)
# config = InferenceConfiguration(confidence_threshold=0.2)

# # initialize the client
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="aCfWpAkGRSWAsm9Suz2u"
# )

# # Configure the client with your settings
# CLIENT.configure(config)

# # Now run inference with these settings
# result = CLIENT.infer(input_image, model_id="walldetector2/1")
# print(result)


# import supervision as sv
# import cv2

# image = cv2.imread(input_image)
# detections = sv.Detections.from_inference(result)

# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# annotated_image = box_annotator.annotate(
#     scene=image, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections)

# sv.plot_image(image=annotated_image, size=(16, 16))

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import supervision as sv
import cv2

# Input image
input_image = "sample_images/test_wall.png"

# Define your custom confidence threshold
config = InferenceConfiguration(confidence_threshold=0.05)

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="aCfWpAkGRSWAsm9Suz2u"
)

# Configure the client with your settings
CLIENT.configure(config)

# Run inference
result = CLIENT.infer(input_image, model_id="walldetector2/1")

# Load image
image = cv2.imread(input_image)

# Create detections
detections = sv.Detections.from_inference(result)

# Count objects
object_counts = {}
for class_id in detections.class_id:
    class_name = result['predictions'][class_id]['class']
    object_counts[class_name] = object_counts.get(class_name, 0) + 1

# Annotate the image
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# Display counts on image
y_offset = 30
for obj, count in object_counts.items():
    cv2.putText(annotated_image, f"{obj}: {count}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    y_offset += 30

# Display the final image
sv.plot_image(image=annotated_image, size=(16, 16))
