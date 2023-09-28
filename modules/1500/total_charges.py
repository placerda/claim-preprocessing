from util.post_processing import sort_words, count_words_in_line
import logging
import re

def extract_total_charges(text):
    formatted_text = ''
    if len(text) >= 3:
        # keep only numbers and
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
    line_threshold = 0.2
    confidence = field['cropping']['confidence']
    
    # iniatialization
    record = {}
    word_count = 0
    buffer = ''
    previous_top = 0
    
    # read words
    words = field['analysis']['words']
    words = sort_words(words, line_threshold)
    for word in words:

        word_content = word['content']
        word_count += 1
        top = word['polygon'][1]
        distance_to_previous = abs(top - previous_top)

        # remove 1 when it is a separator
        words_in_line = count_words_in_line(words, 1, line_threshold)
        if words_in_line >=3 and word_count == 1 and word_content[0] == '1':
            word_content = word_content[1:]

        # get just the first
        if distance_to_previous < line_threshold or previous_top == 0:
            buffer += word_content
            previous_top = top 
        else:
            charge = extract_total_charges(buffer)
            if len(charge) > 3:
                record[f'total_charges_1'] = charge
                logging.info(f'total_charges_1 ({round(confidence,2)}) = {charge}')
            break

    return record