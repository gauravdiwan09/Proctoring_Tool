import base64
from proctoring.proctoring import get_analysis, yolov3_model_v3_path

# Function to convert an image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image

# insert the path of yolov3 model [mandatory]
yolov3_model_v3_path("proctoring/pose_model/yolov3.weights")

# insert the image of base64 format
image_path = "images/single.jpg"
imgData = image_to_base64(image_path)
proctorData = get_analysis(imgData, "proctoring/pose_model/shape_predictor_68_face_landmarks.dat")
print(proctorData)
