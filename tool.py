import torch
import cv2
from PIL import Image
import numpy as np
import base64
from base64 import b64decode
from io import BytesIO
import dlib

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s')

# Initialize face landmarks model
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Replace with the actual path

# Initialize eye aspect ratio detector
ear_threshold = 0.2  # Adjust as needed

# Load MobileNet SSD model for phone detection
phone_net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt", "models/MobileNetSSD_deploy.caffemodel")

# Initialize counters
eye_movements_count = {"looking_up": 0, "looking_down": 0, "looking_left": 0, "looking_right": 0}
head_posture_count = {"looking_up": 0, "looking_down": 0, "looking_left": 0, "looking_right": 0}

def get_eye_aspect_ratio(shape, eye):
    # Define the indexes of the facial landmarks for the left and right eyes
    left_eye_indexes = list(range(36, 42))
    right_eye_indexes = list(range(42, 48))

    # Extract the (x, y) coordinates of the left and right eye landmarks
    left_eye_coords = [(shape.part(i).x, shape.part(i).y) for i in left_eye_indexes]
    right_eye_coords = [(shape.part(i).x, shape.part(i).y) for i in right_eye_indexes]

    # Calculate the Euclidean distances between the vertical eye landmarks
    left_eye_vertical_dist = np.linalg.norm(np.array(left_eye_coords[1]) - np.array(left_eye_coords[5]))
    right_eye_vertical_dist = np.linalg.norm(np.array(right_eye_coords[1]) - np.array(right_eye_coords[5]))

    # Calculate the Euclidean distance between the horizontal eye landmarks
    left_eye_horizontal_dist = np.linalg.norm(np.array(left_eye_coords[0]) - np.array(left_eye_coords[3]))
    right_eye_horizontal_dist = np.linalg.norm(np.array(right_eye_coords[0]) - np.array(right_eye_coords[3]))

    # Calculate the eye aspect ratio (EAR)
    left_ear = left_eye_vertical_dist / left_eye_horizontal_dist
    right_ear = right_eye_vertical_dist / right_eye_horizontal_dist

    # Return the EAR based on the specified eye
    if eye == "left_eye":
        return left_ear
    elif eye == "right_eye":
        return right_ear
    else:
        raise ValueError("Invalid eye specified. Use 'left_eye' or 'right_eye'.")

def base64_to_image(base64_str):
    img_data = b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img = np.array(img)
    return img

def detect_person(img):
    # Process YOLOv5 results and return True if at least one person is detected, False otherwise
    results = model(img)

    # Extract class labels and confidence scores
    class_labels = results.names
    confidences = results.xyxy[0][:, 4]
    class_ids = results.xyxy[0][:, 5]

    # Check if person class is present with confidence above a certain threshold
    person_class_id = class_labels.index('person') if 'person' in class_labels else -1
    confidence_threshold = 0.5  # Adjust as needed

    person_detected = any(confidences[(class_ids == person_class_id)] > confidence_threshold)
    return person_detected


def detect_phone(img):
    # Preprocess the image for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the blob as input to the network
    phone_net.setInput(blob)

    # Forward pass to get the detection
    detections = phone_net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If confidence is above a certain threshold (adjust as needed)
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])

            # Check if the detected object is a phone
            if class_id == 77:  # Update this based on your model
                return True

    # If no phone is detected
    return False

# ... (rest of the code remains the same)
# ... (continued from the previous code snippet)

def detect_eye_movements(shape):
    # Implement eye movement detection using the eye aspect ratio
    ear_left = get_eye_aspect_ratio(shape, "left_eye")
    ear_right = get_eye_aspect_ratio(shape, "right_eye")
    avg_ear = (ear_left + ear_right) / 2

    # Define eye movement directions based on EAR
    if ear_left < avg_ear - 0.05:
        eye_movements_count["looking_left"] += 1
        return "looking_left"
    elif ear_right < avg_ear - 0.05:
        eye_movements_count["looking_right"] += 1
        return "looking_right"
    elif ear_left > avg_ear + 0.05 and ear_right > avg_ear + 0.05:
        eye_movements_count["looking_up"] += 1
        return "looking_up"
    elif ear_left < avg_ear - 0.05 and ear_right < avg_ear - 0.05:
        eye_movements_count["looking_down"] += 1
        return "looking_down"
    else:
        return "normal"

# ... (previous code remains the same)
# ... (continued from the previous code snippet)

def detect_head_posture_movements(shape):
    # Get facial landmarks
    left_eye = list(shape.parts())[36:42]
    right_eye = list(shape.parts())[42:48]
    nose = list(shape.parts())[27:36]

    # Calculate the center of each eye and the midpoint of the line connecting the eyes
    left_eye_center = np.mean([(point.x, point.y) for point in left_eye], axis=0).astype("int")
    right_eye_center = np.mean([(point.x, point.y) for point in right_eye], axis=0).astype("int")
    eyes_midpoint = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

    # Calculate the slope of the line passing through the eyes
    eyes_slope = (right_eye_center[1] - left_eye_center[1]) / (right_eye_center[0] - left_eye_center[0] + 1e-10)

    # Calculate the slope of the line passing through the eyes and nose
    eyes_nose_slope = (eyes_midpoint[1] - nose[8].y) / (eyes_midpoint[0] - nose[8].x + 1e-10)

    # Determine head posture based on slopes
    if eyes_slope > 0.5:
        head_posture_count["looking_right"] += 1
        return "looking_right"
    elif eyes_slope < -0.5:
        head_posture_count["looking_left"] += 1
        return "looking_left"
    elif eyes_nose_slope > 0.5:
        head_posture_count["looking_up"] += 1
        return "looking_up"
    elif eyes_nose_slope < -0.5:
        head_posture_count["looking_down"] += 1
        return "looking_down"
    else:
        return "normal"

# ... (rest of the code remains the same)
# ... (continued from the previous code snippet)


# Main function to process the image
def process_image(base64_str):
    img = base64_to_image(base64_str)

    # Detect person
    person_detected = detect_person(img)

    # Detect phone
    phone_detected = detect_phone(img)

    # Detect eye movements
    shape = shape_predictor(img, dlib.rectangle(0, 0, img.shape[1], img.shape[0]))
    eye_movements = detect_eye_movements(shape)

    # Detect head posture movements
    head_posture_movements = detect_head_posture_movements(shape)

    return {
        "person_detected": "more_than_one_person_detected" if person_detected else "normal",
        "phone_detected": "mobile_phone_detected" if phone_detected else "no_mobile_phone_detected",
        "eye_movements": eye_movements,
        "head_posture_movements": head_posture_movements,
        "eye_movements_count": eye_movements_count,
        "head_posture_count": head_posture_count,
        # Add more keys for additional functionalities
    }

# Read image file
with open("images/camera.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Process the image
result = process_image(base64_image)
print(result)