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
        return None

def run(field):

    record = {}
    line_threshold = 0.2
    word_count = 0
    buffer = ''
    birth_date = ''
    confidence = field['cropping']['confidence']
    if confidence < 0:
        return record

    # read words
    words = field['analysis']['words']
    words = sort_words(words, line_threshold)
    words =  [word for word in words if len([char for char in word['content'] if char.isdigit()]) >= 2]

    for word in words:

        word_count += 1
        word_content = ''.join([char for char in word['content'] if char.isdigit()])

        # remove 1 when it is a separator
        if len(word_content) in (3, 5, 7, 9):
            if word_content[0] == '1':
                word_content = word_content[1:]
            elif word_content[-1] == '1':    
                word_content = word_content[:-1]
        if len(word_content) == 5 and word_content[2] == '1':
                word_content = word_content[:2] + word_content[3:]
                
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

        if word_count == len(words): # last word
            birth_date = buffer
            if len(buffer) <= 8 and count_digits(buffer) >= 6:
                birth_date = format_date(buffer)        
                record[f'birth_date_1'] = birth_date
                logging.info(f'birth_date_1 ({round(confidence,2)}) = {birth_date}')
    
    return record
