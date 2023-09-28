from util.post_processing import sort_words, count_words_in_line
import logging
import re

def extract_charges(text):
    formatted_text = ''
    
    if len(text) >= 3:
        # keep only numbers
        text = re.sub(r"[^0-9]", "", text)
        # put in format 0.00
        if len(text) == 2:
            formatted_text = '0.' + text
        else:
            formatted_text = text[:-2] + '.' + text[-2:]
    
    elif len(text) == 2:
        text = re.sub(r"[^0-9]", "", text)
        formatted_text = '0.' + text
    
    return formatted_text

def run(field):

    # parameters
    max_distance_between_rows = 1.5
    line_threshold = 0.2
    line_number = 1

    # initialize variables
    record = {}
    confidence = field['cropping']['confidence']
    word_position = 0
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
        if(distance_to_previous > line_threshold and  previous_top > 0 and line_number < 7) :
            buffer = extract_charges(buffer)
            if len(buffer) > 0:
                record[f'charges_{line_number}'] = buffer
                logging.info(f'charges_{line_number} ({round(confidence,2)}) = {buffer}')
                line_number += 1                                                 
                buffer=''
        
        if distance_to_previous >  max_distance_between_rows: 
            break

         # remove 1 when it is a separator
        word_content = word['content']
        words_in_line = count_words_in_line(words, line_number, line_threshold)
        if words_in_line > 1 and word_position == words_in_line and len(word_content) == 3:
            if word_content[0] == '1':
                word_content = word_content[1:]
        if words_in_line== 3 and word_position == 2:
            if word_content[0] == '1':
                word_content = word_content[1:]

        # append word to buffer
        buffer = buffer +  word_content
        previous_top = top     

        # process last row
        if (word_count == len(words) and line_number < 7): 
            buffer = extract_charges(buffer)
            if len(buffer) > 0:
                record[f'charges_{line_number}'] = buffer
                logging.info(f'charges_{line_number} ({round(confidence,2)}) = {buffer}')
                line_number += 1                                                 
                buffer=''        

    return record



            
