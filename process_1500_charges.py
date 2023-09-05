from util.utils import load_image, get_filename
from util.pre_processing import remove_blobs, remove_hlines, crop_charges, crop_total_charges
from util.post_processing import extract_charges, sort_words, count_words_in_line
from util.computervision_api import object_detection_rest
from util.formrec_api import analyze_document_rest
import cv2
import csv
from datetime import datetime
from glob import glob
import logging
import numpy as npc
import os
from dotenv import load_dotenv
import argparse

load_dotenv()
VISION_MODEL = os.environ.get("VISION_MODEL")

# logging level
logging.basicConfig(level=logging.INFO)

def process_forms(files=None):

    # folders
    work_dir = 'work'
    default_input = 'data/test/*.pdf' # when no input is provided

    # define input files
    if files is None:
        files = glob(default_input)
    elif files.endswith('.pdf'):
        files = [files]
    elif files.endswith('/'):
        files = glob(f'{files}*.pdf')
    else:
        files = glob(f'{files}/*.pdf')
    
    results = []
    if len(results) == 0:
        logging.info(f"No files to process")

    ###############################
    # extract charges
    ###############################

    for idx, image_file in enumerate(files):

        prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        logging.info(f"### PROCESSING FILE: {image_file} ({prefix})")

        if not os.path.exists(image_file):
            logging.info(f"File not found: {image_file}")
            continue

        record = {
            'fileName': image_file.split('/')[-1],
            'charges_1': '', 
            'charges_2': '', 
            'charges_3': '',
            'charges_4': '',
            'charges_5': '',
            'charges_6': '',
            'total_charges': ''
        }

        ###############################
        # read input file
        ###############################    

        input = load_image(image_file, prefix, prefix='input', width=1700)
        input_filename = get_filename(prefix+'_'+image_file.split('/')[-1].split('.')[0], "input")
        cv2.imwrite(input_filename, input)

        ###############################
        # run object detection
        ###############################

        object_detection_result = object_detection_rest(input_filename, VISION_MODEL)

        ###############################
        # extract charges grid
        ###############################

        # crop charges
        cropped_charges, confidence, found_charges = crop_charges(input_filename, object_detection_result) # type: ignore
        if not found_charges:
            logging.info(f"Could not detect charges. Confidence: {confidence}")
            results.append(record)

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
        max_distance_between_charges = 200
        first_page = fr_result['pages'][0]
        line_number = 0
        word_position = 0
        previous_top = 0
        word_count = 0
        buffer = ''
        # read words
        words = first_page['words']
        words = sort_words(words, line_threshold)
        for word in words:  
            word_count += 1
            top = word['polygon'][1] # top
            if abs(top - previous_top) > line_threshold and line_number < 6:
                if previous_top > 0:
                    charge = extract_charges(buffer)
                    if len(charge) > 3:
                        line_number += 1
                        record[f'charges_{line_number}'] = charge
                        logging.info(f'charges_{line_number} ({round(confidence,2)}) = {charge}')
                    word_position = 0                
                    buffer = ''
            word_content = word['content']
            word_position += 1
            previous_top = top


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
                if line_number < 6 and len(charge) > 3:
                    record[f'charges_{line_number}'] = charge
                    logging.info(f'charges_{line_number} ({round(confidence,2)})  = {charge}')

        ###############################
        # extract total charges
        ###############################

        cropped_total_charges, confidence, found_charges = crop_total_charges(input_filename, object_detection_result) # type: ignore
        if not found_charges:
            logging.info(f"Could not detect total charges. Confidence: {confidence}")
            results.append(record)

        cropped_total_charges = cv2.cvtColor(cropped_total_charges, cv2.COLOR_BGR2GRAY)
        cropped_filename = get_filename(prefix, "cropped_total_charges")
        cv2.imwrite(cropped_filename, cropped_total_charges)

        cleaned_total_charges = remove_blobs(cropped_total_charges, 40)
        cleaned_total_charges = remove_hlines(cleaned_total_charges)
        cleaned_total_charges = remove_blobs(cleaned_total_charges, 10)
        cleaned_total_charges = remove_hlines(cleaned_total_charges)

        total_charges_cleaned_filename = get_filename(prefix, "total_charges_cleaned" )
        cv2.imwrite(total_charges_cleaned_filename, cleaned_total_charges) 

        fr_result = analyze_document_rest(total_charges_cleaned_filename, "prebuilt-layout", features=['ocr.highResolution'])
        line_threshold = 30
        first_page = fr_result['pages'][0]
        line_number = 0
        word_position = 0
        previous_top = 0
        word_count = 0
        buffer = ''

        # read words
        words = first_page['words']
        words = sort_words(words, line_threshold)
        for word in words:

            word_count += 1
            top = word['polygon'][1] # top
            if abs(top - previous_top) > line_threshold and line_number <= 1:
                if previous_top > 0:
                    charge = extract_charges(buffer)
                    if len(charge) > 3:
                        line_number += 1
                        record[f'total_charges'] = charge
                        logging.info(f'total_charges ({round(confidence,2)}) = {charge}')
                        break
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
                if line_number <= 1 and len(charge) > 3:
                    record[f'total_charges'] = charge
                    logging.info(f'total_charges ({round(confidence,2)}) = {charge}')
                    break
                
        results.append(record)
            
    if len(results) > 0:
        # save results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(work_dir): os.makedirs(work_dir)
        output_file = os.path.join(work_dir, f'{timestamp}.csv')

        # assuming you have a list of dictionaries called `results`
        header = ["fileName", "charges_1", "charges_2", "charges_3", "charges_4", "charges_5", "charges_6", "total_charges"]
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        logging.info(f"Results file: {output_file}") 
        
    logging.info(f"Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract form 1500 fields.')
    parser.add_argument('-i', '--input', help='Folder where the pdfs are or a single pdf file.')
    args = parser.parse_args()

    if args.input:
        process_forms(args.input)
    else:
        process_forms()