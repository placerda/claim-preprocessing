from util.post_processing import sort_words, count_words_in_line
import logging
import re

def extract_qty(text):
    # keep only integer part
    if '.' in text:
        text = text.split('.')[0]
    # keep only numbers    
    text = re.sub(r"[^0-9]", "", text)
    return text


def run(field):
    
    # parameters
    confidence = field['cropping']['confidence']
    word_position_in_row = 0
    max_distance_between_rows = 0.9
    line_threshold = 0.06

    # initialize variables
    record = {}
    line_number = 1
    previous_top = 0
    last_record_top = 0    
    word_count = 0
    buffer = ''

    # get words
    words = field['analysis']['words']
    words = sort_words(words, line_threshold)

    for word in words:
        word_count += 1
        word_position_in_row += 1        
        top = word['polygon'][1] # top
        distance_to_previous = abs(top - previous_top)
        distance_to_last_record = abs(top - last_record_top)  

        # process row
        if (distance_to_previous > line_threshold and  previous_top > 0 and line_number < 7): 
            buffer = extract_qty(buffer)
            if len(buffer) > 0:
                record[f'qty_{line_number}'] = buffer
                logging.info(f'qty_{line_number} ({round(confidence,2)}) = {buffer}')
                word_position_in_row = 1
                last_record_top = previous_top                
                line_number += 1                                                 
                buffer=''

        if distance_to_last_record >  max_distance_between_rows: 
            break

        # keep only integer part
        word_content = word['content']
        # example: words = [1 00 => 1]
        if word_position_in_row > 1:
            word_content = ''

        # append word to buffer
        buffer += word_content
        previous_top = top     
            
        # process last row
        if (word_count == len(words) and line_number < 7): 
            buffer = extract_qty(buffer)
            if len(buffer) > 0:
                record[f'qty_{line_number}'] = buffer
                logging.info(f'qty_{line_number} ({round(confidence,2)}) = {buffer}')
                line_number += 1                                                 
                buffer=''            

    return record