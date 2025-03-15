roboflow_models = {
    "walls": {
        "model_id": "walldetector2/1",
        "confidence_threshold": 0.1,
        "label_colors": {"walls": (0, 128, 255)},  # Blue
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
        "label_colors": {"doors": (0, 0, 255) },  # Red
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
        "confidence_threshold": 0.3,
        "label_colors": {
            "sink": (0, 255, 0),  # Green
            "bathroom": (255, 0, 0),  # Orange
            # "door": (0, 0, 255),  # Red
            "toilet": (128, 0, 128)  # Purple
        },
        "comments": "7/10 result for toilets",
        "display": True,
    },
}

# Define the Roboflow API URL
API_URL = "https://detect.roboflow.com"
API_KEY = "aCfWpAkGRSWAsm9Suz2u"
