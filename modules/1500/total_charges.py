from util.post_processing import sort_words, count_words_in_line
import logging
import re

def extract_total_charges(text):
    formatted_text = ''
    # keep only numbers
    text = re.sub(r"[^0-9]", "", text)
    if len(text) >= 3:
        # put in format 0.00
        if len(text) == 2:
            formatted_text = '0.' + text
        else:
            formatted_text = text[:-2] + '.' + text[-2:]
    elif len(text) == 2:
        formatted_text = '0.' + text
    return formatted_text

def run(field):

    # parameters
    line_threshold = 0.06
    confidence = field['cropping']['confidence']
    
    # iniatialization
    record = {}
    word_count = 0
    word_position_in_row = 0
    buffer = ''
    previous_top = 0
    
    # read words
    words = field['analysis']['words']
    words = sort_words(words, line_threshold)
    for word in words:

        word_content = word['content']
        word_position_in_row += 1
        word_count += 1
        top = word['polygon'][1]
        distance_to_previous = abs(top - previous_top)

        # remove 1 when it is a separator
        words_in_line = count_words_in_line(words, 1, line_threshold)
        # example: words = [$ 99 100 => $ 99 00]
        if words_in_line > 1 and word_position_in_row == words_in_line and len(word_content) == 3:
            if word_content.startswith('1'):
                word_content = word_content[1:]

        # for this field we're interested just in the first row
        if distance_to_previous < line_threshold or previous_top == 0:
            buffer += word_content
            previous_top = top 
        else:
            charge = extract_total_charges(buffer)
            if len(charge) > 3:
                record[f'total_charges_1'] = charge
                logging.info(f'total_charges_1 ({round(confidence,2)}) = {charge}')
            break

    # process last row (for cases where first and last row are the same)
    if (word_count == len(words)): 
        charge = extract_total_charges(buffer)
        if len(charge) > 3:
            record[f'total_charges_1'] = charge
            logging.info(f'total_charges_1 ({round(confidence,2)}) = {charge}')    

    return record