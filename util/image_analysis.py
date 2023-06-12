# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
import cv2

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
def get_document_tables(document_path):
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
    return tables

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