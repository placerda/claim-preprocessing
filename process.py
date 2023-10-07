from dotenv import load_dotenv
from PIL import Image
from util.general import load_image, get_filename
from util.pre_processing import crop, remove_blobs, remove_hlines
from util.computervision_api import object_detection_rest
from util.formrec_api import analyze_document_rest
from datetime import datetime
from glob import glob
import argparse
import cv2
import csv
import importlib
import os
import logging
import numpy as np
import os
import yaml

# logging levelF
logging.basicConfig(level=logging.INFO)
load_dotenv()
VISION_MODEL = os.environ.get("VISION_MODEL")
PAGE_WIDTH=1700
PAGE_HEIGHT=2256

def main(config_file, files=None):
    
    #####################
    # Initialization
    #####################

    work_dir = 'work'
    # define input

    default_input = 'data/*.pdf'
    if files is None:
        files = glob(default_input)
    elif files.endswith('.pdf'):
        files = [files]
    elif files.endswith('/'):
        files = glob(f'{files}*.pdf')
    else:
        files = glob(f'{files}/*.pdf')
    
    # load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if len(files) == 0:
        logging.info(f"No files to process")
        exit(0)

    #####################
    # Process each file
    #####################

    for idx, image_file in enumerate(files):

        results = []

        prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        logging.info(f"### PROCESSING FILE: {image_file} ({prefix})")

        # initialize record with all fields values empty
        record = {'fileName': image_file.split('/')[-1] }
        header = ['fileName']
        for field in config['fields']:
            for i in range(1, field['cardinality']+1):
                field_name = f"{field['name']}_{i}"
                record[field_name] = ''
                header.append(field_name)

        # read input file
        input = load_image(image_file, prefix, prefix='input', width=PAGE_WIDTH)
        # looking only at the first page
        input_filename = get_filename(prefix+'_'+image_file.split('/')[-1].split('.')[0], "input")
        cv2.imwrite(input_filename, input)

        ##########################
        # Preprocessing
        ##########################

        # Detection
        
        object_detection_result = object_detection_rest(input_filename, VISION_MODEL)

        fields = config['fields']

        for field in fields:

            # Cropping
            cropped, confidence, found = crop(input_filename, object_detection_result, field['cropping'])
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

            cropped_filename = get_filename(prefix, f"cropped_{field['name']}")
            cv2.imwrite(cropped_filename, cropped)
            field['cropping']['filename'] = cropped_filename

        # Save cropped images to combined pdf

        images = []
        page_number = 1
        for field in fields:
            images.append(Image.fromarray(field['cropping']['roi']))
            field['cropping']['page_number'] = page_number
            page_number += 1
        combined_file = f"{work_dir}/{prefix}-combined.pdf"       
        with open(combined_file, "wb") as pdf_file:
            images[0].save(pdf_file, "PDF", save_all=True, append_images=images[1:], resolution=200, optimize=True, quality=100)

        #####################
        # Document Analysis
        #####################
        
        fr_result = analyze_document_rest(combined_file, config['document_analysis']['model'], config['document_analysis']['api_version'], []) # , ['ocr.highResolution']

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
        results.append(record)

        #####################
        # Save results
        #####################
        
        if len(results) > 0:
            output_file = os.path.join(work_dir, f'{prefix}.csv')
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
            logging.info(f"Results file: {output_file}") 
          
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