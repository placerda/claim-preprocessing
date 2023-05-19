# APPROACH 02 - ALIGNMENT > APPLY FIELDS MASK > OCR

# -----------------------------
#   USAGE
# -----------------------------
# python preprocessing_02.py --labels_mask labels_mask.jpg --values_mask values_mask.jpg --template template.jpg --image image.jpg

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
from image_alignment import align
from image_analysis import get_document_fields

# load environment variables
dotenv.load_dotenv()

def main(args):
    print(f"[INFO] Starting script.")

    # get input filenames
    work_image = args["image"]
    work_template = args["template"]
    work_values_mask = args["values_mask"]    
    work_labels_mask = args["labels_mask"]
    # log filenames
    print(f"[INFO] Input image: {work_image}")
    print(f"[INFO] Template: {work_template}")
    print(f"[INFO] Values Mask: {work_values_mask}")
    print(f"[INFO] Labels Mask: {work_labels_mask}")

    # define working files
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # timestamp prefix
    work_image = get_work_filename(timestamp, 'input', work_image)
    work_template = get_work_filename(timestamp, 'template', work_template)    
    work_values_mask = get_work_filename(timestamp, 'values_mask', work_values_mask)
    work_labels_mask = get_work_filename(timestamp, 'labels_mask', work_labels_mask)
    print(f"[INFO] Work image: {work_image}")
    print(f"[INFO] Work template: {work_template}")
    print(f"[INFO] Values mask template: {work_values_mask}")
    print(f"[INFO] Labels mask template: {work_labels_mask}")

    # Load the input image and template
    print("[INFO] Loading work images...")
    image = cv2.imread(work_image)
    template = cv2.imread(work_template)
    values_mask = cv2.imread(work_values_mask, cv2.IMREAD_GRAYSCALE)
    labels_mask = cv2.imread(work_labels_mask, cv2.IMREAD_GRAYSCALE)    
    width = 1700
    print(f"[INFO] Resizing to {width} width.")
    image = imutils.resize(image, width=width)
    template = imutils.resize(template, width=width)
    values_mask = imutils.resize(values_mask, width=width)
    labels_mask = imutils.resize(labels_mask, width=width)

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

    # Apply masks
    print("[INFO] Applying masks...")
    # masked = cv2.bitwise_and(aligned, aligned, mask=mask)
    aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    masked = np.where(values_mask == 0, aligned, 255)
    masked = np.where(labels_mask < 255, labels_mask, masked)
    masked_image = get_work_filename(timestamp, 'masked')
    cv2.imwrite(masked_image, masked)
    print(f"[INFO] Masked image: {masked_image}")

    # Extract values with document pre_built model
    print("[INFO] Extracting values with OCR...")
    fields = get_document_fields(masked_image)
    output_text = get_work_filename(timestamp, 'output', extension='json')
    with open(output_text, 'w') as f:
        json.dump(fields, f, indent=4)
    print(f"[INFO] Output saved to {output_text}")
    
    print('[INFO] Done')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to input image that we'll align to template")
    ap.add_argument("-t", "--template", required=True, help="Path to input template image")
    ap.add_argument("-l", "--labels_mask", required=True, help="Path to labels mask")
    ap.add_argument("-v", "--values_mask", required=True, help="Path to values mask")    
    args = vars(ap.parse_args())    
    main(args)