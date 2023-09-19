from util.utils import load_image, get_filename
from util.pre_processing import remove_blobs, remove_hlines, crop_qty, crop_dates, crop_charges, crop_total_charges
from util.post_processing import extract_charges, sort_words, count_words_in_line, extract_qty
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
    # process forms
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
            'qty_1': '', 
            'qty_2': '', 
            'qty_3': '',
            'qty_4': '',
            'qty_5': '',
            'qty_6': '',            
            'total_charges': '',
            'start_date_1': '', 
            'start_date_2': '', 
            'start_date_3': '',
            'start_date_4': '',
            'start_date_5': '',
            'start_date_6': '',   
            'end_date_1': '', 
            'end_date_2': '', 
            'end_date_3': '',
            'end_date_4': '',
            'end_date_5': '',
            'end_date_6': ''
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
        # extract charges
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

        fr_result = analyze_document_rest(charges_cleaned_filename, "prebuilt-read")  # features=['ocr.highResolution']
        line_threshold = 25
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
            if abs(top - previous_top) > line_threshold and line_number < 7:
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

            buffer += word_content
            if word_count == len(words): # last word
                line_number += 1
                charge = extract_charges(buffer)
                if line_number < 7 and len(charge) > 3:
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

        fr_result = analyze_document_rest(total_charges_cleaned_filename, "prebuilt-read") # features=['ocr.highResolution']
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

            buffer += word_content
            if word_count == len(words): # last word
                line_number += 1
                charge = extract_charges(buffer)
                if line_number <= 1 and len(charge) > 3:
                    record[f'total_charges'] = charge
                    logging.info(f'total_charges ({round(confidence,2)}) = {charge}')
                    break

        ###############################
        # extract dates
        ###############################

        def count_digits(string):
            count = 0
            for char in string:
                if char.isdigit():
                    count += 1
            return count
        
        def format_date(date_str):
            if len(date_str) == 6:
                return f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"
            elif len(date_str) == 8:
                return f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"
            else:
                return None

        # crop dates
        cropped_dates, confidence, found_dates = crop_dates(input_filename, object_detection_result) # type: ignore
        if not found_dates:
            logging.info(f"Could not detect dates. Confidence: {confidence}")
            results.append(record)

        cropped_dates = cv2.cvtColor(cropped_dates, cv2.COLOR_BGR2GRAY)
        cropped_filename = get_filename(prefix, "cropped_dates")
        cv2.imwrite(cropped_filename, cropped_dates)

        cleaned_dates = remove_blobs(cropped_dates, 40)
        cleaned_dates = remove_hlines(cleaned_dates)
        cleaned_dates = remove_blobs(cleaned_dates, 10)
        cleaned_dates = remove_hlines(cleaned_dates)

        dates_cleaned_filename = get_filename(prefix, "dates_cleaned" )
        cv2.imwrite(dates_cleaned_filename, cleaned_dates) 

        fr_result = analyze_document_rest(dates_cleaned_filename, "prebuilt-read")  # features=['ocr.highResolution']
        line_threshold = 20
        first_page = fr_result['pages'][0]
        line_number = 0
        word_position = 0
        previous_top = 0
        word_count = 0
        buffer = ''
        start_date=''
        end_date=''        
        # read words
        words = first_page['words']
        words = sort_words(words, line_threshold)

        words =  [word for word in words if len([char for char in word['content'] if char.isdigit()]) >= 2]

        for word in words:
            word_count += 1
            word_content = ''.join([char for char in word['content'] if char.isdigit()])

            if len(word_content) > 1 and len(word_content) <= 8 and count_digits(word_content) >= 2:
                top = word['polygon'][1] # top
                if abs(top - previous_top) > line_threshold and line_number < 7:
                    if previous_top > 0:
                        # date = buffer # extract_dates(buffer)
                        if len(start_date) >=6 :
                            line_number += 1
                            start_date = format_date(start_date)
                            end_date = format_date(end_date)                             
                            record[f'start_date_{line_number}'] = start_date
                            logging.info(f'start_date_{line_number} ({round(confidence,2)}) = {start_date}')
                            record[f'end_date_{line_number}'] = end_date
                            logging.info(f'end_date_{line_number} ({round(confidence,2)}) = {end_date}')                        
                        word_position = 0                
                        buffer = ''
                        start_date=''
                        end_date=''

                word_position += 1
                previous_top = top

                # post-processing to remove 1's that are actually pipes (review)
                words_in_line = count_words_in_line(words, line_number, line_threshold)
                if len(word_content) == 3 or len(word_content) == 5:
                    if word_content[0] == '1':
                        word_content = word_content[1:]
                    elif word_content[-1] == '1':    
                        word_content = word_content[:-1]

                if len(start_date) < 6:
                    start_date = start_date +  word_content
                else:
                    end_date = end_date + word_content

                if word_count == len(words): # last word
                    line_number += 1
                    # date = date #  extract_charges(buffer)
                    if line_number < 7 and len(start_date) >= 6:
                        start_date = format_date(start_date)
                        end_date = format_date(end_date)                        
                        record[f'start_date_{line_number}'] = start_date
                        logging.info(f'start_date_{line_number} ({round(confidence,2)}) = {start_date}')
                        record[f'end_date_{line_number}'] = end_date
                        logging.info(f'end_date_{line_number} ({round(confidence,2)}) = {end_date}')                          
                
    
        ###############################
        # extract qty
        ###############################

        # crop qty
        cropped_qty, confidence, found_qty = crop_qty(input_filename, object_detection_result) # type: ignore
        if not found_qty:
            logging.info(f"Could not detect qty. Confidence: {confidence}")
            results.append(record)

        cropped_qty = cv2.cvtColor(cropped_qty, cv2.COLOR_BGR2GRAY)
        cropped_filename = get_filename(prefix, "cropped_qty")
        cv2.imwrite(cropped_filename, cropped_qty)

        cleaned_qty = remove_blobs(cropped_qty, 40)
        cleaned_qty = remove_hlines(cleaned_qty)
        cleaned_qty = remove_blobs(cleaned_qty, 10)
        cleaned_qty = remove_hlines(cleaned_qty)

        qty_cleaned_filename = get_filename(prefix, "qty_cleaned" )
        cv2.imwrite(qty_cleaned_filename, cleaned_qty) 

        fr_result = analyze_document_rest(qty_cleaned_filename, "prebuilt-read")  # features=['ocr.highResolution']
        line_threshold = 25
        max_distance_between_qty = 200
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
            if abs(top - previous_top) > line_threshold and line_number < 7:
                if previous_top > 0:
                    qty = extract_qty(buffer)
                    if len(qty) > 0:
                        line_number += 1
                        record[f'qty_{line_number}'] = qty
                        logging.info(f'qty_{line_number} ({round(confidence,2)}) = {qty}')
                    word_position = 0                
                    buffer = ''
            word_content = word['content']
            word_position += 1
            previous_top = top

            if word_position == 1:
                buffer += word_content
                
            if word_count == len(words): # last word
                line_number += 1
                qty = extract_qty(buffer)
                if line_number < 7 and len(qty) > 0:
                    record[f'qty_{line_number}'] = qty
                    logging.info(f'qty_{line_number} ({round(confidence,2)}) = {qty}')

        ###############################
        # extract birth date
        ###############################
        # TODO


        results.append(record)
            
    if len(results) > 0:
        # save results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(work_dir): os.makedirs(work_dir)
        output_file = os.path.join(work_dir, f'{timestamp}.csv')

        # assuming you have a list of dictionaries called `results`
        header = ['fileName', 'charges_1', 'charges_2', 'charges_3', 'charges_4', 'charges_5', 'charges_6', 'total_charges',
                  'qty_1', 'qty_2', 'qty_3', 'qty_4', 'qty_5', 'qty_6',
                  'start_date_1', 'start_date_2','start_date_3','start_date_4', 'start_date_5', 'start_date_6', 'end_date_1',
                  'end_date_2', 'end_date_3', 'end_date_4', 'end_date_5', 'end_date_6']

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