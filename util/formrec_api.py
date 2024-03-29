# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from requests.exceptions import ConnectionError
import azure.ai.vision as sdk
import os
import cv2
import base64
import requests
import json
import time

# globals
# FORM_REC_API_VERSION = "2023-07-31" or "2023-02-28-preview"
VISION_ENDPOINT=os.environ["VISION_ENDPOINT"]
VISION_KEY = os.environ["VISION_KEY"]
QR_CODE_MODEL_NAME = "qrcode01"
CHARGES_MODEL_NAME = "charges01"

# -----------------------------
#   FUNCTIONS
# -----------------------------

### Form Recognize Service functions

def analyze_document_sdk(filepath: str, model: str):
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

''' parse tables in the json result '''
def get_tables_and_paragraphs(document_path):
    data = analyze_document_sdk(document_path, 'prebuilt-layout')
    tables = []
    for idx, table in enumerate(data.tables):
        print(f"[INFO] Parsing table {str(idx+1).zfill(3)}")
        nodes = []
        for cell in table.cells:
            rowIndex = cell.row_index
            columnIndex = cell.column_index
            node = {}
            node['content'] = cell.content
            node['row'] = cell.row_index
            node['column'] = cell.column_index
            nodes.append(node)
        tables.append(nodes)
    paragraphs = []
    print(f"[INFO] Parsing paragraphs")
    for idx, paragraph in enumerate(data.paragraphs):
        item = {}
        item['content'] = paragraph.content
        item['top'] = paragraph.bounding_regions[0].polygon[0].y
        paragraphs.append(item)
    return tables, paragraphs

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

def get_base64_encoded_content(bytes):
    return base64.b64encode(bytes).decode("utf-8")

def get_base64_encoded_file_content(filepath):
        input_bytes = open(filepath, "rb").read()
        return get_base64_encoded_content(input_bytes)

def analyze_document_rest(image_data, model, api_version, features=[]):

    # Input used to be a file read as bytes before we changed to inmemory approach.
    # base64EncodedContent = get_base64_encoded_content(image_data)
    base64EncodedContent = get_base64_encoded_content(image_data)

    # Request headers
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": os.environ['FORM_RECOGNIZER_KEY']
    }

    # Request body
    body = {
        "base64Source": base64EncodedContent
    }

    # if user wants to get features
    if len(features) > 0:
        features_str = ",".join(features)    
        request_endpoint = f"{os.environ['FORM_RECOGNIZER_ENDPOINT']}formrecognizer/documentModels/{model}:analyze?api-version={api_version}&features={features_str}"
    else:
        request_endpoint = f"{os.environ['FORM_RECOGNIZER_ENDPOINT']}formrecognizer/documentModels/{model}:analyze?api-version={api_version}"
    
    try:
        # Send request
        response = requests.post(request_endpoint, headers=headers, json=body)
    except requests.exceptions.ConnectionError as e:
        print("[INFO] Connection error, retrying in 10seconds...")
        time.sleep(10)
        response = requests.post(request_endpoint, headers=headers, json=body)

    # Parse response
    if response.status_code == 202:
        # Request accepted, get operation ID
        operation_id = response.headers["Operation-Location"].split("/")[-1]
        # print("Operation ID:", operation_id)
    else:
        # Request failed
        print("[formrec_api] Error request: ", response.text)
        exit()

    # Poll for result
    result_endpoint = f"{os.environ['FORM_RECOGNIZER_ENDPOINT']}formrecognizer/documentModels/{model}/analyzeResults/{operation_id}"
    result_headers = headers.copy()
    result_headers["Content-Type"] = "application/json-patch+json"
    result = {}

    while True:
        result_response = requests.get(result_endpoint, headers=result_headers)
        result_json = json.loads(result_response.text)

        if result_response.status_code != 200 or result_json["status"] == "failed":
            # Request failed
            print("Error result: ", result_response.text)
            break

        if result_json["status"] == "succeeded":
            # Request succeeded, print result
            # print("Result:", json.dumps(json.dumps(result_json['analyzeResult']), indent=4))
            result = result_json['analyzeResult']
            break

        # Request still processing, wait and try again
        time.sleep(2)

    return result

# some fields have their value (content) split in two or more lines and we need to concatenate them
def concatenate_lines(selected_line, lines, max_dist=20):
    selected_polygon = selected_line['polygon']
    y = selected_polygon[1]
    # polygon coords: [x1,y1,x2,y2,x3,y3,x4,y4]
    polygon = selected_polygon.copy()
    content = ""
    for line in lines:
        if (y-max_dist < line['polygon'][1] < y+max_dist):
            # content
            content = content + " " + line['content'].strip()
            # polygon x's
            if line['polygon'][0] < polygon[0]:
                polygon[0] = line['polygon'][0]
                polygon[6] = line['polygon'][0]
            if line['polygon'][2] > polygon[2]:
                polygon[2] = line['polygon'][2]
                polygon[4] = line['polygon'][2]
            # polygon y's
            if line['polygon'][1] < polygon[1]:
                polygon[1] = line['polygon'][1]
                polygon[3] = line['polygon'][1]
            if line['polygon'][5] > polygon[5]:
                polygon[5] = line['polygon'][5]
                polygon[7] = line['polygon'][5]                
    return content.strip(), polygon