# APPROACH 01 - ALIGNMENT > SHIFTING > REMOVING VERTICAL LINES

# -----------------------------
#   USAGE
# -----------------------------
# python preprocessing_01.py --template template.jpg --image image.jpg

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import argparse
import cv2
import datetime
import dotenv
import imutils
import json
from general_utilities import get_work_filename
from general_utilities import write_image
from general_utilities import write_debug_image
from image_alignment import align
from image_processing import shift_values
from image_processing import remove_vertical_lines
from image_analysis import crop_from_qrcode, get_document_fields, get_document_words
from image_analysis import remove_separators

# load environment variables
dotenv.load_dotenv()

def main(args):
    print(f"[INFO] Starting script.")

    # get input filenames
    work_image = args["image"]
    work_template = args["template"]
    # log filenames
    print(f"[INFO] Input image: {work_image}")
    print(f"[INFO] Template: {work_template}")

    # define working files
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # timestamp prefix
    work_image = get_work_filename(timestamp, 'input', work_image)
    work_template = get_work_filename(timestamp, 'template', work_template)    
    print(f"[INFO] Work image: {work_image}")
    print(f"[INFO] Work template: {work_template}")

    # Crop image from qr code to the bottom of the image
    # print("[INFO] Cropping work image...")
    # work_image = get_work_filename(timestamp, 'cropped', work_image)  
    # cropped = crop_from_qrcode(work_image)
    # cv2.imwrite(work_image, cropped)
    # print(f"[INFO] Cropped image: {work_image}")

    # Load the input image and template
    print("[INFO] Loading work images...")
    image = cv2.imread(work_image)
    template = cv2.imread(work_template)
    width = 1700
    print(f"[INFO] Resizing to {width} width.")
    image = imutils.resize(image, width=width)
    template = imutils.resize(template, width=width)

    # Align image with template
    print("[INFO] Aligning images...")
    
    aligned = align(image, template, debug=False) 
    aligned_image = get_work_filename(timestamp, 'aligned') # aligned image
    cv2.imwrite(aligned_image, aligned)
    print(f"[INFO] Aligned image: {aligned_image}")

    overlay = template.copy()
    output = aligned.copy()
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    overlay_image = get_work_filename(timestamp, 'overlay')
    cv2.imwrite(overlay_image, output)
    print(f"[INFO] Overlay image: {overlay_image}")

    # Shifting values down
    print('[INFO] Shifting values down')
    shifted_debug = shift_values(aligned, template)
    shifted_debug = cv2.convertScaleAbs(shifted_debug)
    shifted_debug_image = get_work_filename(timestamp, 'shifted_debug')
    cv2.imwrite(shifted_debug_image, shifted_debug)
    shifted = cv2.cvtColor(shifted_debug, cv2.COLOR_BGR2GRAY)  
    shifted = cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR)      
    shifted_image = get_work_filename(timestamp, 'shifted')
    cv2.imwrite(shifted_image, shifted)   
    print(f"[INFO] Shifted image: {shifted_image}")

    # Remove vertical lines (separators)
    print("[INFO] Removing vertical lines (separators)...")
    
    # Object detection approach 
    # noseparators = remove_separators(aligned_image)
    # noseparators_image = get_work_filename(timestamp, 'noseparators')
    # cv2.imwrite(noseparators_image, noseparators)
    # write_debug_image(noseparators, width) # debug
    # print(f"[INFO] No separators image: {noseparators_image}")

    # Computer vision approach
    words = get_document_words(shifted_image)
    noseparators, noseparators_debug = remove_vertical_lines(shifted, words)
    noseparators_image = get_work_filename(timestamp, 'noseparators')
    cv2.imwrite(noseparators_image, noseparators)
    noseparators_debug_image = get_work_filename(timestamp, 'noseparators_debug')
    cv2.imwrite(noseparators_debug_image, noseparators_debug)
    print(f"[INFO] No separators image: {noseparators_image}")
    
    # Output file
    output = cv2.cvtColor(noseparators, cv2.COLOR_BGR2GRAY)
    output_image = get_work_filename(timestamp, 'output')
    write_image(output_image, output)
    print(f"[INFO] Output image: {output_image}")

    # Extract values with document pre_built model
    print("[INFO] Extracting values with OCR...")
    fields = get_document_fields(output_image)
    output_text = get_work_filename(timestamp, 'output', extension='json')
    with open(output_text, 'w') as f:
        json.dump(fields, f, indent=4)
    print(f"[INFO] Output saved to {output_text}")

    print('[INFO] Done')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to input image that we'll align to template")
    ap.add_argument("-t", "--template", required=True, help="Path to input template image")
    args = vars(ap.parse_args())    
    main(args)