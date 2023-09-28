from util.post_processing import sort_words
import logging

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
        return ''

def run(field):
    
    # parameters
    confidence = field['cropping']['confidence']
    name = field['name']
    max_distance_between_rows = 1.5
    line_threshold = 0.2

    record = {}

    # initialize row_processing variables
    line_number = 1
    previous_top = 0
    word_count = 0
    buffer = ''
    skipped_last_row = False
    skip_row = False
    
    # get words (digits only)
    words = field['analysis']['words']
    words = sort_words(words, line_threshold)
    words =  [word for word in words if len([char for char in word['content'] if char.isdigit()]) >= 2]

    for word in words:
        word_count += 1
        top = word['polygon'][1] # top

        if skipped_last_row:
            distance_to_previous = 0
            skipped_last_row = False
        else:
            distance_to_previous = top - previous_top
            
        # process row
        if (distance_to_previous > line_threshold and  previous_top > 0 and line_number < 7): 
            buffer = format_date(buffer)
            if ((count_digits(buffer) == 6 or count_digits(buffer) == 8)) and not skip_row:
                record[f'{name}_{line_number}'] = buffer                
                logging.info(f'{name}_{line_number} ({round(confidence,2)}) = {buffer}')     
                line_number += 1                                              
                buffer=''
                skipped_last_row = False
            else: 
                skipped_last_row = True


        if distance_to_previous >  max_distance_between_rows: 
            break

        # remove 1's that are separators
        word_content = word['content']
        if len(word_content) in (3, 5, 7, 9):
            if word_content[0] == '1' and word_content[-1].isdigit():
                word_content = word_content[1:]
            elif word_content[-1] == '1' and word_content[0].isdigit():    
                word_content = word_content[:-1]
        if len(word_content) == 5 and word_content[2] == '1' and word_content[0].isdigit():
                word_content = word_content[:2] + word_content[3:]

        # remove any no n digit characters
        word_content = ''.join([char for char in word_content if char.isdigit()])

        # append word to buffer
        if len(word_content) in (2, 4, 6, 8):
            buffer = buffer +  word_content
            previous_top = top     

        # process last row
        if (word_count == len(words) and line_number < 7): 
            buffer = format_date(buffer)
            if ((count_digits(buffer) == 6 or count_digits(buffer) == 8) and line_number < 7):
                record[f'{name}_{line_number}'] = buffer                
                logging.info(f'{name}_{line_number} ({round(confidence,2)}) = {buffer}')     
                line_number += 1               

    return record