from util.post_processing import sort_words, count_words_in_line
import logging
import re

def extract_qty(text):
    if '.' in text:
        text = text.split('.')[0]    
    text = re.sub(r"[^0-9]", "", text)
    return text


def run(field):
    
    # parameters
    confidence = field['cropping']['confidence']
    max_distance_between_rows = 1.5
    line_threshold = 0.2

    # initialize variables
    record = {}
    line_number = 1
    previous_top = 0
    word_count = 0
    buffer = ''

    # get words
    words = field['analysis']['words']
    words = sort_words(words, line_threshold)

    for word in words:
        word_count += 1
        top = word['polygon'][1] # top
        distance_to_previous = abs(top - previous_top)

        # process row
        if (distance_to_previous > line_threshold and  previous_top > 0 and line_number < 7): 
            buffer = extract_qty(buffer)
            if len(buffer) > 0:
                record[f'qty_{line_number}'] = buffer
                logging.info(f'qty_{line_number} ({round(confidence,2)}) = {buffer}')
                line_number += 1                                                 
                buffer=''

        if distance_to_previous >  max_distance_between_rows: 
            break

        # append word to buffer
        word_content = word['content']
        if len(word_content) == 2 or len(word_content) == 4:
            buffer = buffer +  word_content
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