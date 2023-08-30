import argparse
from util.utils import load_image, get_filename
import cv2
from datetime import datetime
from glob import glob
import logging
import os
import imutils
from dotenv import load_dotenv
from pdf2image import convert_from_path
load_dotenv()

# logging level
logging.basicConfig(level=logging.INFO)

# initialization

def main(input_path):
    logging.info(f"Starting...")

    # check if input_path is a file or a folder
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = glob(input_path + '/*.pdf')
    
    for idx, input_file in enumerate(files):

        prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        logging.info(f"### PROCESSING FILE: {input_file} ({prefix})")
        output_file = input_file.replace('.pdf', '.jpg')

        pages = convert_from_path(input_file, dpi=200)
        pages[0].save(output_file, 'JPEG')
        image = cv2.imread(output_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = imutils.resize(image, width=1700)
        cv2.imwrite(output_file, image)

    logging.info(f"Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PDF form files to JPEGs.')
    parser.add_argument('input_path', type=str, help='Path to the input PDF file or folder containing PDF files.')
    args = parser.parse_args()
    main(args.input_path)