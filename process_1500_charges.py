from util.general_utilities import load_image, get_filename
from util.image_processing import remove_blobs, remove_hlines
from util.field_extraction import extract_charges
from util.image_analysis import object_detection_rest, analyze_document_rest
import cv2
import csv
from datetime import datetime
from glob import glob
import logging
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()


# logging level
logging.basicConfig(level=logging.INFO)

# initialization
work_dir = 'work'
# files = glob('data/test/*.pdf')
# files = glob('data/newtest/*.pdf')
files = ['data/newtest/19018150.pdf']
results = []

###############################
# extract charges
###############################

def crop_charges(input_image):

    # cropping parameters
    roi_max_height = 400
    side_border = 7 # pixels
    height_multiplier = 7
    min_charges_detection_confidence = 0.7

    # initialize the cropped images
    cropped_charges = np.zeros((roi_max_height,400,3),np.uint8)
    confidence = -1.0
    found_charges = False
    result = object_detection_rest(input_image, 'charges01')

    for object in result['customModelResult']['objectsResult']['values']:
        for tag in object['tags']:
            if tag['confidence'] >= min_charges_detection_confidence and tag['name'] == 'charges':
                image = cv2.imread(input_image)
                x = object['boundingBox']['x']
                y = object['boundingBox']['y']
                w = object['boundingBox']['w']
                h = object['boundingBox']['h']

                roi_height = height_multiplier*h
                roi_height = min(roi_height, roi_max_height)
                cropped_charges = image[y+h:y+roi_height, x+side_border:x+w-side_border]
              
                confidence = tag['confidence']
                found_charges = True

    return cropped_charges, confidence, found_charges

def count_words_in_line(words, line_number, line_threshold):
    word_count = 0
    previous_top = 0
    current_line = -1
    line_threshold = 5 # adjust this value to fit your needs
    for word in words:
        top = word['polygon'][1]
        if abs(top - previous_top) > line_threshold:
            if current_line == line_number:
                return word_count            
            current_line += 1
            previous_top = top
            word_count = 0
        word_count += 1
    # last line
    if current_line == line_number:
        return word_count
    return 0

# def count_words_in_line(words, line_number, line_threshold):
#     word_count = 0
#     previous_top = 0
#     current_line = -1
#     for word in words:
#         word_count += 1
#         top = word['polygon'][1]
#         if abs(top - previous_top) > line_threshold:
#             if current_line == line_number:
#                 return word_count            
#             current_line += 1
#             previous_top = top
#             word_count = 0
#     # last line
#     return word_count
        

def sort_words(words):
    words = sorted(words, key=lambda word: (word['polygon'][1]))
    rows = []
    row = []

    for word in words:
        if len(row) == 0:
            row.append(word)
        else:
            if abs(word['polygon'][1] - row[0]['polygon'][1]) < line_threshold:
                row.append(word)
            else:
                rows.append(row)
                row = []
                row.append(word)
    if len(row) > 0:
        rows.append(row)
    
    words = []
    for row in rows:
        row = sorted(row, key=lambda word: (word['polygon'][0]))
        words.append(row)

    # flatten
    words = [item for sublist in words for item in sublist]
    
    return words

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
        logging.info(f"Could not detect charges. Confidence: {confidence}")
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
    line_threshold = 30
    first_page = fr_result['pages'][0]
    line_number = 0
    word_position = 0
    previous_top = 0
    word_count = 0
    buffer = ''
    # read words
    words = first_page['words']
    words = sort_words(words)
    for word in words:  
        word_count += 1
        top = word['polygon'][1] # top
        if abs(top - previous_top) > line_threshold and line_number <= 6:
            if previous_top > 0:
                charge = extract_charges(buffer)
                if len(charge) > 3:
                    line_number += 1
                    record[f'charges_{line_number}'] = charge
                    logging.info(f'charges_{line_number} = {charge}')
                word_position = 0                
                buffer = ''
        previous_top = top
        word_content = word['content']
        word_position += 1

        # post-processing to remove 1's that are actually pipes (review)
        words_in_line = count_words_in_line(words, line_number, line_threshold)
        if words_in_line > 1 and word_position == words_in_line and len(word_content) == 3:
            if word_content[0] == '1':
                word_content = word_content[1:]
            elif word_content[2] == '1':    
                word_content = word_content[:2]
        if words_in_line == 1 and word_position == words_in_line and len(word_content) == 3:
            if word_content[0] == '1':
                word_content = word_content[1:]
            elif word_content[2] == '1':    
                word_content = word_content[:2]

        # post-processing to add integral part when missing
        if words_in_line == 1 and len(word_content) == 2:
                word_content = '0.' + word_content

        buffer += word_content
        if word_count == len(words): # last word
            line_number += 1
            charge = extract_charges(buffer)
            if line_number <= 6 and len(charge) > 3:
                record[f'charges_{line_number}'] = charge
                logging.info(f'charges_{line_number} = {charge}')

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