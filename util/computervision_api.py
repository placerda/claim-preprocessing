# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import os
import cv2
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# globals
FORM_REC_API_VERSION = "2023-07-31"
VISION_ENDPOINT=os.environ["VISION_ENDPOINT"]
VISION_KEY = os.environ["VISION_KEY"]
QR_CODE_MODEL_NAME = "qrcode01"
CHARGES_MODEL_NAME = "charges01"

# -----------------------------
#   FUNCTIONS
# -----------------------------

### Computer Vision Analysis

def object_detection_rest(filepath, model):
    
    # Request headers
    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": VISION_KEY
    }

    image_data = open(filepath, "rb").read()

    request_endpoint = f"{VISION_ENDPOINT}computervision/imageanalysis:analyze?api-version=2023-02-01-preview&model-name={model}"
    
    # Send request
    response = requests.post(request_endpoint, headers=headers, data=image_data)

    # Parse response
    if response.status_code in (200, 202):
        result = json.loads(response.text)
    else:
        # Request failed
        print("Error request: ", response.text)
        exit()

    return result

def crop_from_qrcode(input_image):
    """
    Crops an image to the bounding box and bellow of a QR code detected in the image using Computer Vision service.

    Args:
        input_image (str): The path to the input image file.

    Returns:
        Optional[np.ndarray]: The cropped image as a NumPy array, or None if no QR code was found.
    """
    service_options = sdk.VisionServiceOptions(os.environ["VISION_ENDPOINT"], os.environ["VISION_KEY"])
    vision_source = sdk.VisionSource(filename=input_image)
    analysis_options = sdk.ImageAnalysisOptions()
    analysis_options.model_name = QR_CODE_MODEL_NAME

    # do the analysis
    image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)
    result = image_analyzer.analyze()

    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:
        if result.custom_objects is not None:
            for object in result.custom_objects:
                if object.confidence > 0.8 and object.name == "qrcode":
                    # print("[INFO] found '{}', {} Confidence: {:.4f}".format(object.name, object.bounding_box, object.confidence))
                    image = cv2.imread(input_image)
                    x = object.bounding_box.x
                    y = object.bounding_box.y
                    height, width = image.shape[:2]
                    qr_width = object.bounding_box.w
                    qr_height = object.bounding_box.h
                    qr_area = qr_width*qr_height
                    # ideal_width = int(round(qr_width*qrcode_to_width_ratio))
                    # ideal_height = int(round(qr_height*qrcode_to_height_ratio))
                    # cropped_image = image[y:min(y+ideal_height, height), x:min(x+ideal_width, width)]
                    cropped_image = image[y:, x:]                    
                    return cropped_image, object.confidence, qr_area
        else:
            error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
            print("[ERROR] Analysis failed.")
            print("            Error reason: {}".format(error_details.reason))
            print("            Error code: {}".format(error_details.error_code))
            print("            Error message: {}".format(error_details.message))
