import re
import string
from datetime import datetime, timedelta
from util.openai_api import complete
from util.utils import count_digits

def remove_non_alphanumeric(text):
    pattern = r'[^a-zA-Z0-9\s]+'
    return re.sub(pattern, '', text)

def extract_insured_id(text):
    # clean removing all non-alphanumeric but spaces
    text = remove_non_alphanumeric(text)
    # print(f"[INFO] Insured id cleaned: {text}")
    # return id with 8-12 digits
    pattern = r'\b[a-zA-Z]?\d{8,12}\b'
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        return ""

def extract_cpthcpccode(text):
    # clean removing all non-alphanumeric but spaces
    text = remove_non_alphanumeric(text)
    # print(f"[INFO]  Cpt cleaned: {text}")
    # return cpt code
    pattern = r'\b([0-9]{4}[0-9A-Z]{1}|[0-9]{5}|[A-Z][0-9]{4})\b'
    # pattern = r'\b[A-Va-v]?\d{4,5}\b'
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        return ""

def extract_charges(text):
    formatted_text = ''
    if len(text) >= 3:
        # keep only numbers and
        text = re.sub(r"[^0-9]", "", text)
        # put in format 0.00
        formatted_text = text[:-2] + '.' + text[-2:]
    elif len(text) == 2:
        text = re.sub(r"[^0-9]", "", text)
        formatted_text = '0.' + text
    return formatted_text
    
def is_valid_date(date_string, max_years_old=5):
    try:
        date = datetime.strptime(date_string, '%m/%d/%Y')
        if date <= datetime.today() and date >= datetime.today() - timedelta(days=(365*max_years_old)):
            return True
        else:
            return False
    except ValueError:
        try:
            # obs: does not check if the year is in the past when using 2 digits
            date = datetime.strptime(date_string, '%m/%d/%y')
            if date <= datetime.today() and date >= datetime.today() - timedelta(days=(365*max_years_old)):
                return True
            else:
                return False
        except ValueError:
            return False

def format_date(input_str):
    # Remove everything that is not a number or space
    input_str = re.sub(r'[^0-9\s]', '', input_str)
    
    # Remove spaces
    input_str = input_str.replace(' ', '')
    
    # Format date based on length
    if len(input_str) == 6:
        return f"{input_str[:2]}/{input_str[2:4]}/{input_str[4:]}"
    elif len(input_str) == 8:
        return f"{input_str[:2]}/{input_str[2:4]}/{input_str[4:]}"
    elif len(input_str) == 5:
        return f"0{input_str[:1]}/{input_str[-4:-2]}/{input_str[-2:]}"
    else:
        return input_str    
    
def extract_date(text):
    
    # cleaning common noisy characters
    text = text.replace(':', ' ')
    text = text.replace('!', ' ')
    text = text.replace('_', ' ')
    
    # remove all other punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # replace multiple consective spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # remove noisy from field labels (birth date field)
    text = text.replace("3 P", "P")
    text = text.replace("3P", "P")
    text = text.replace("6 P", "P")
    text = text.replace("6P", "P")    
    parts = text.split("PATIENT REL")
    text = parts[0]
    text = re.sub(r'[^\d\s]+', '', text).strip()

    # print(f"[INFO] Date regex: {text}") 
    text = format_date(text)

    return text


def llm_extract_date(text):
    prompt_filename = "prompts/date_inference.txt"
    generated_date = complete(prompt_filename, text).strip()
    # keep only numbers and /
    pattern = r"[0-9/]+"
    matches = re.findall(pattern, generated_date)
    generated_date =  "".join(matches)
    if generated_date == "":
        generated_date = "could not infer"
    else:
        if count_digits(generated_date) == 8 and count_digits(text) == 6:
            # force year to two digits when the input has only 6 digits
            # prompt instruction to keep input format is not working
            generated_date = generated_date[:-4] + generated_date[-2:]
    return generated_date

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

def sort_words(words, line_threshold):
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
