# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import azure.ai.vision as sdk
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------

QR_CODE_MODEL_NAME = "qrcode01"
SEPARATORS_MODEL_NAME = "separators04"

def crop_from_qrcode(input_image: str):
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
                    print("[INFO] found '{}', {} Confidence: {:.4f}".format(object.name, object.bounding_box, object.confidence))
                    image = cv2.imread(input_image)
                    y = object.bounding_box.y
                    height, width = image.shape[:2]
                    cropped_image = image[y:height, :]
                    return cropped_image
        else:
            error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
            print("[ERROR] Analysis failed.")
            print("            Error reason: {}".format(error_details.reason))
            print("            Error code: {}".format(error_details.error_code))
            print("            Error message: {}".format(error_details.message))

def remove_separators(image_path, separator_color):
    """
    Removes any horizontal or vertical separators from an image using Computer Vision service.

    Args:
        image_path (str): The path to the input image file.
        separator_color (Tuple[int, int, int]): The RGB color of the separators to remove.

    Returns:
        Image: The modified image with separators removed.
    """
    service_options = sdk.VisionServiceOptions(os.environ["VISION_ENDPOINT"], os.environ["VISION_KEY"])
    vision_source = sdk.VisionSource(filename=input_image)
    analysis_options = sdk.ImageAnalysisOptions()
    analysis_options.model_name = SEPARATORS_MODEL_NAME
    
    # do the analysis
    image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)
    result = image_analyzer.analyze()
    result_image = cv2.imread(input_image)
    
    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:
        if result.custom_objects is not None:
            for object in result.custom_objects:
                if object.confidence > 0.2 and object.name == "separator":
                    print("[INFO] found '{}', {} Confidence: {:.4f}".format(object.name, object.bounding_box, object.confidence))
                    x0 = object.bounding_box.x
                    y0 = object.bounding_box.y
                    x1 = object.bounding_box.x + object.bounding_box.w
                    y1 = object.bounding_box.x + object.bounding_box.h
                    cv2.rectangle(result_image, (x0,y0), (x1,y1), (255, 255, 255), -1)

        else:
            error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
            print("[ERROR] Analysis failed.")
            print("            Error reason: {}".format(error_details.reason))
            print("            Error code: {}".format(error_details.error_code))
            print("            Error message: {}".format(error_details.message))

    return result_image

def analyze_document_sdk(filepath: str, model: str):
    """
    Analyzes a document image using the Azure Form Recognizer service.

    Args:
        filepath (str): The path to the document image file.
        model (str): The ID of the Form Recognizer model to use for analysis.

    Returns:
        DocumentAnalysis: The result of the document analysis.

    Raises:
        ValueError: If the specified file path is invalid or the analysis fails.
    """
    document_analysis_client = DocumentAnalysisClient(
        endpoint=os.environ['FORM_RECOGNIZER_ENDPOINT'], credential=AzureKeyCredential(os.environ['FORM_RECOGNIZER_KEY'])
    )

    with open(filepath, "rb") as f:
        poller = document_analysis_client.begin_analyze_document(
            model, document=f
        )
    result = poller.result()

    return result


def get_document_fields(document_path):
    result = {}
    analysis = analyze_document_sdk(document_path, 'prebuilt-document')
    fields = analysis.key_value_pairs
    for field in fields:
        if field.value is not None:
            result[field.key.content] = field.value.content
    return result

def get_document_words(document_path):
    result = []
    analysis = analyze_document_sdk(document_path, 'prebuilt-read')
    words = analysis.pages[0].words
    for word in words:
        # bbox format: (x, y, w, h)
        bbox = (int(word.polygon[0].x),
                int(word.polygon[0].y),
                int(word.polygon[1].x - word.polygon[0].x),
                int(word.polygon[2].y - word.polygon[0].y))
        result.append(bbox)
    return result