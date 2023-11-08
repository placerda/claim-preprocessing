# Python Standard Library Imports
import argparse
import csv
import logging
import os
from datetime import datetime
from glob import glob
from io import BytesIO

# Third Party Imports
import cv2
import importlib
import imutils
import numpy as np
import yaml
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

# Local Imports
from util.computervision_api import object_detection_rest
from util.formrec_api import analyze_document_rest
from util.general import get_filename
from util.pre_processing import crop, remove_blobs, remove_hlines

# Constants
LOGGING_LEVEL = logging.INFO
VISION_MODEL_ENV_VAR = "VISION_MODEL"
DEBUG_MODE_ENV_VAR = "DEBUG_MODE"
DEBUG_MODE_DEFAULT = 'false'
PAGE_WIDTH = 1700
PAGE_HEIGHT = 2256
WORK_DIR = 'work'
DEFAULT_INPUT = 'data/*.pdf'

# Configure Logging
logging.basicConfig(level=LOGGING_LEVEL)
load_dotenv()

VISION_MODEL = os.environ.get(VISION_MODEL_ENV_VAR)
DEBUG_MODE = os.environ.get(DEBUG_MODE_ENV_VAR) or DEBUG_MODE_DEFAULT
debug_mode = True if DEBUG_MODE.lower() == 'true' else False


def get_files(files):
    if files is None:
        return glob(DEFAULT_INPUT)
    elif files.endswith('.pdf'):
        return [files]
    elif files.endswith('/'):
        return glob(f'{files}*.pdf')
    else:
        return glob(f'{files}/*.pdf')

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def process_forms(files, config):
    """
    Processes a list of forms based on the provided configuration.

    This function iterates over a list of form files, processes each form using the provided configuration, and writes the results to a CSV file. 
    The CSV file is named with the current timestamp and stored in a predefined working directory.

    Parameters:
    files (list of str): A list of file paths to the forms that need to be processed.
    config (dict): A dictionary containing configuration options for processing. 
                    The 'fields' key should contain a list of dictionaries, each representing a field to be processed in the form. 
                    Each field dictionary should have a 'name' key (the name of the field) and a 'cardinality' key (the number of times the field appears in the form).

    Returns:
    None

    Raises:
    - FileNotFoundError: If any of the form files in the list do not exist.
    - KeyError: If the config dictionary does not contain the required keys.
    """    
    # create empty output file with header
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(WORK_DIR, f'{timestamp}.csv')
    logging.info(f"### PROCESSING START ({output_file})")  
        
    header = ['fileName']
    for field in config['fields']:
        for i in range(1, field['cardinality']+1):
            field_name = f"{field['name']}_{i}"
            header.append(field_name)

    if not os.path.exists(WORK_DIR): os.makedirs(WORK_DIR)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
  
    # Process each file
    for idx, image_file in enumerate(files):
        record = process_form(image_file, config)
        with open(output_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writerow(record)

    logging.info(f"### PROCESSING DONE ({output_file})")  

def initialize_record(config, image_file):
    record = {'fileName': image_file.split('/')[-1]}
    for field in config['fields']:
        for i in range(1, field['cardinality']+1):
            field_name = f"{field['name']}_{i}"
            record[field_name] = ''
    return record

def process_form(form_file, config):
    """
    Processes a single form based on the provided configuration.

    This function reads a form file, applies object detection to identify fields, crops the fields, removes noise, applies document analysis, and finally post-processes the results.

    Parameters:
    form_file (str): The path to the form file that needs to be processed.
    config (dict): A dictionary containing configuration options for processing. The 'fields' key should contain a list of dictionaries, each representing a field to be processed in the form. Each field dictionary should have a 'name' key (the name of the field), a 'cropping' key (parameters for cropping the field), a 'remove_noise' key (a boolean indicating whether noise should be removed from the field), and a 'postprocessing' key (parameters for post-processing the field).

    Returns:
    record (dict): A dictionary containing the extracted field values.

    The function works as follows:
    - It first initializes a record for the form.
    - It then reads the form file and resizes the image.
    - It applies object detection to the image.
    - It iterates over each field in the config:
        - It crops the field from the image.
        - If the 'remove_noise' key is true, it removes noise from the field.
        - It adds a white border to the cropped field.
    - It saves the cropped fields to a combined PDF.
    - It applies document analysis to the PDF.
    - It post-processes the results of the document analysis.
    - Finally, it returns the record with the extracted field values.
    """

    logging.info(f"### PROCESSING FILE: {form_file}")

    record = initialize_record(config, form_file)

    # read input file
    pages = convert_from_path(form_file, dpi=200, first_page=0, last_page=1)
    image = np.array(pages[0])
    input_image = imutils.resize(image, width=PAGE_WIDTH)

    if debug_mode:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        input_filename = get_filename(timestamp+'_'+form_file.split('/')[-1].split('.')[0], "input")
        cv2.imwrite(input_filename, input_image) 
    
    success, encoded_image = cv2.imencode('.jpg', input_image)
    input_bytes = encoded_image.tobytes()

    ##########################
    # Preprocessing
    ##########################

    # Detection
    object_detection_result = object_detection_rest(input_bytes, VISION_MODEL)

    fields = config['fields']

    for field in fields:

        # Cropping
        cropped, confidence, found = crop(input_image, object_detection_result, field['cropping'])
        field['cropping']['confidence'] = confidence
        field['cropping']['found'] = found
        if not found:
            logging.info(f"Could not detect {field['name']}. Confidence: {confidence}")
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # Remove noise
        if field['remove_noise']:
            cropped = remove_blobs(cropped, 40)
            cropped = remove_hlines(cropped)
            cropped = remove_blobs(cropped, 10)
            cropped = remove_hlines(cropped)
            
        # Add white border to cropped image
        border_size = 10
        cropped = cv2.copyMakeBorder(cropped, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        field['cropping']['roi'] = cropped

        if debug_mode:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            cropped_filename = get_filename(timestamp, f"cropped_{field['name']}")
            cv2.imwrite(cropped_filename, cropped)
            field['cropping']['filename'] = cropped_filename

    # Save cropped images to combined pdf
    images = []
    page_number = 1

    for field in fields:
        images.append(Image.fromarray(field['cropping']['roi']))
        field['cropping']['page_number'] = page_number
        page_number += 1
    
    buffer = BytesIO()
    images[0].save(buffer, "PDF", save_all=True, append_images=images[1:], resolution=200, optimize=True, quality=100)
    buffer.seek(0)
    pdf_data = buffer.read()

    #####################
    # Document Analysis
    #####################
    
    fr_result = analyze_document_rest(pdf_data, config['document_analysis']['model'], config['document_analysis']['api_version'], []) # , ['ocr.highResolution']

    #####################
    # Postprocessing 
    #####################

    for field in fields:
        for page in fr_result['pages']:
            if page['pageNumber'] == field['cropping']['page_number']:
                field['analysis'] = {'words': page['words']} 
                module = importlib.import_module("modules." + field['postprocessing']['module'])
                post_process_result = module.run(field)
                record = {**record, **post_process_result}
                break

    return record
        
def main(config_file, files=None):
    files = get_files(files)
    config = load_config(config_file)

    if len(files) == 0:
        logging.info(f"No files to process")
        exit(0)

    process_forms(files, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract form fields.')
    parser.add_argument('-i', '--input', help='Folder where the pdfs are or a single pdf file.')
    parser.add_argument('-c', '--config', help='Document intelligence config file.')    
    args = parser.parse_args()

    if args.input:
        main(args.config, args.input)
    else:
        main(args.config)

    logging.info(f"Done")