from util.general_utilities import load_image, get_filename, count_digits
from util.image_analysis import crop_from_qrcode, analyze_document_rest, concatenate_lines
from util.image_processing import extract_roi, remove_blobs, remove_hlines
from util.field_extraction import extract_insured_id, extract_date, is_valid_date, llm_extract_date, extract_cpthcpccode, extract_charges
import azure.ai.vision as sdk
import cv2
import csv
from datetime import datetime
from glob import glob
import imutils
import logging
import numpy as np
import os

# logging level

logging.basicConfig(level=logging.INFO)

# initialization
work_dir = 'work'

files = glob('data/p1500-analysis/*.pdf')
# files = glob('data/test/*.pdf')
# files = ['data/test/18943432_0.pdf']
results = []

###############################
# extract charges
###############################

def crop_charges(input_image):
    service_options = sdk.VisionServiceOptions(os.environ["VISION_ENDPOINT"], os.environ["VISION_KEY"])
    vision_source = sdk.VisionSource(filename=input_image)
    analysis_options = sdk.ImageAnalysisOptions()
    analysis_options.model_name = 'charges01'

    # initialize the cropped images
    cropped_charges = np.zeros((400,400,3),np.uint8)
    confidence = 0.0
    found_charges = False

    # do the analysis
    image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)
    result = image_analyzer.analyze()

    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:
        if result.custom_objects is not None:
            for object in result.custom_objects:
                if object.confidence > 0.7 and object.name == "charges":
                    # print("[INFO] found '{}', {} Confidence: {:.4f}".format(object.name, object.bounding_box, object.confidence))
                    image = cv2.imread(input_image)
                    x = object.bounding_box.x
                    y = object.bounding_box.y
                    w = object.bounding_box.w
                    h = object.bounding_box.h                    
                    cropped_charges = image[y+h:y+(7*h)-10, x+5:x+w-10]
                    confidence = object.confidence
                    found_charges = True
        else:
            error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
            print("[ERROR] Analysis failed.")
            print("            Error reason: {}".format(error_details.reason))
            print("            Error code: {}".format(error_details.error_code))
            print("            Error message: {}".format(error_details.message))

    return cropped_charges, confidence, found_charges


for idx, image_file in enumerate(files):

    prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"### PROCESSING FILE: {image_file} ({prefix})")

    record = {
        'fileName': image_file.split('/')[-1],
        'charges_1': '', 
        'charges_2': '', 
        'charges_3': '',
        'charges_4': '',
        'charges_5': '',
        'charges_6': ''
    }

    ###############################
    # read input files and do the cropping
    ###############################    

    input = load_image(image_file, prefix, prefix='input', width=1700)
    input_filename = get_filename(prefix+'_'+image_file.split('/')[-1].split('.')[0], "input")
    cv2.imwrite(input_filename, input)

    ###############################
    # extract charges
    ###############################

    # crop charges
    cropped_charges, confidence, found_charges = crop_charges(input_filename) # type: ignore
    if not found_charges:
        logging.info(f"Low confidence record not processed.")
        results.append(record)
        continue

    cropped_charges = cv2.cvtColor(cropped_charges, cv2.COLOR_BGR2GRAY)
    cropped_filename = get_filename(prefix, "cropped_charges")
    cv2.imwrite(cropped_filename, cropped_charges)

    cleaned_charges = remove_blobs(cropped_charges, 40)
    cleaned_charges = remove_hlines(cleaned_charges)
    cleaned_charges = remove_blobs(cleaned_charges, 10)
    cleaned_charges = remove_hlines(cleaned_charges)

    charges_cleaned_filename = get_filename(prefix, "charges_cleaned" )
    cv2.imwrite(charges_cleaned_filename, cleaned_charges)
    fr_result = analyze_document_rest(charges_cleaned_filename, "prebuilt-layout", features=['ocr.highResolution'])
    i = 1
    for paragraph in fr_result['paragraphs']:
        charge = paragraph['content'].strip()
        charge_extracted = extract_charges(charge)
        if len(charge_extracted) < 4: continue
        record[f'charges_{i}'] = charge_extracted
        logging.info(f"Charge {i} extracted: {charge_extracted}")
        i += 1

    ###############################
    # TODO: extract total charges
    ###############################

    results.append(record)
        
# save results
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(work_dir): os.makedirs(work_dir)
output_file = os.path.join(work_dir, f'{timestamp}.csv')

# assuming you have a list of dictionaries called `results`
header = ["fileName", "charges_1", "charges_2", "charges_3", "charges_4", "charges_5", "charges_6"]

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    for result in results:
        writer.writerow(result)
logging.info(f"Results file: {output_file}") 
logging.info(f"Done")