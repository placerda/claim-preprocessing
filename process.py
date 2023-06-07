# APPROACH 03 - 01. ALIGNMENT > 02. APPLY FIELDS MASKS > 03. REMOVE TEMPLATE > 04. ADD TABLE FRAME > 05. EXTRACT TABLES WITH FR LAYOUT/CUSTOM MODEL

# -----------------------------
#   USAGE
# -----------------------------
# python preprocessing_03.py --labels_mask labels_mask.jpg --values_mask values_mask.jpg --tables_frame tables_frame.jpg --template template.jpg --image image.jpg --keep_percent 0.3

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import json
import numpy as np
import argparse
import cv2
import datetime
import dotenv
from util.general_utilities import load_image, get_filename
from util.image_alignment import align
from util.image_processing import remove_template
from util.image_analysis import get_document_tables

# load environment variables
dotenv.load_dotenv()

def calculate_iou(region1, region2):
    # Calculate intersection
    intersection = np.logical_and(region1, region2)
    intersection_area = np.sum(intersection)

    # Calculate union
    union = np.logical_or(region1, region2)
    union_area = np.sum(union)

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def main(args):
    print(f"[INFO] Starting script.")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # timestamp prefix
    print("[INFO] Loading and resizing images...")
    image = load_image(args["image"], timestamp, prefix='input')
    template = load_image(args["template"], timestamp, prefix='template' )    
    values_mask = load_image(args["values_mask"], timestamp, prefix='values_mask', gray=True )
    labels_mask = load_image(args["labels_mask"], timestamp, prefix='labels_mask', gray=True )
    tables_frame = load_image(args["tables_frame"], timestamp, prefix='tables_frame', gray=True )

    ## PREPROCESSING

    # 01. Align image with template (try it n times)
    
    # find best parameter:
    n = 1
    iou_threshold = 0.4
    keep_percentage = 0.3
    increment = 0.1
    best_keep_percentage = 0
    best_iou = 0
    best_index = -1
    aligned_images = []
    print(f"[INFO] 01. Aligning input document ...")
    for i in range(n):
        aligned = align(
            image,
            template,
            maxFeatures=500000,
            keepPercent=keep_percentage,
        )
        aligned_images.append(aligned)

        aligned_gray= cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        aligned_gray = cv2.threshold(aligned_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        aligned_region = aligned_gray[455:854, 1579:1611]
        aligned_region[aligned_region > 0] = 1
        template_gray= cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        template_roi = template_gray[455:854, 1579:1611]
        template_roi[template_roi > 0] = 1
        iou = round(calculate_iou(template_roi, aligned_region),2)

        if iou >= best_iou:
            best_iou = iou
            best_keep_percentage = keep_percentage
            best_index = i

        print(f"[INFO] Trial {str(i+1).zfill(2)} k={keep_percentage} iou={iou}")
        
        keep_percentage += increment
        keep_percentage = round(keep_percentage, 2)

    if best_iou >= iou_threshold:
        print(f"[INFO] Alignment was successful ({best_iou}) keep=({best_keep_percentage}).")
        aligned = aligned_images[best_index]
    else:
        print(f"[INFO] Alignment was not successful ({best_iou}). Please try a different approach.")            
        exit()

    aligned_filename = get_filename(timestamp, "aligned")  # aligned image
    cv2.imwrite(aligned_filename, aligned)
    print(f"[INFO] Aligned image: {aligned_filename}")

    overlay = template.copy()
    output = aligned.copy()
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    overlay_filename = get_filename(timestamp, "overlay" )
    cv2.imwrite(overlay_filename, output)
    print(f"[INFO] Overlay image: {overlay_filename}")

    # 02-03. Apply masks and remove background elements
    print("[INFO] 02-04. Applying masks, removing background and adding tables frames...")
    image_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    # Checking how the input image becomes when removing the background elements
    no_background = image_gray.copy()
    no_background = remove_template(no_background, template)
    no_background_filename = get_filename(timestamp, "no-background" )
    cv2.imwrite(no_background_filename, no_background)

    # values mask
    masked = np.where(values_mask == 0, image_gray, 255)
    
    # labels mask
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    masked = np.where(labels_mask < 255, template_gray, masked)

    # 04. Add Table frame
    tables_light = np.where(tables_frame < 255, 225, 255)
    masked = np.where(tables_frame <255, tables_light, masked)
    
    masked_filename = get_filename(timestamp, "masked" )
    cv2.imwrite(masked_filename, masked)
    print(f"[INFO] Masked image: {masked_filename}")

    # 05. Duplicate charges row to force FR read it as a table
    print("[INFO] 05. Duplicating charges row...")
    
    # get roi
    output = masked.copy()
    x1, y1, x2, y2 = 100, 1755, 1588, 1822 # charges row area
    h = y2 - y1
    w = x2 - x1
    charges_row_roi = masked[y1:y1+h, x1:x1+w]
    
    # clear original charges row
    output[y1:y1+h, x1:x1+w] = 255 
    
    shift_down_pixels = 58
    # first shift
    shift_down = 2 * shift_down_pixels
    output[y1+shift_down:y1+shift_down+h, x1:x1+w] = charges_row_roi
    # second shift (right below the first one)
    shift_down = 3 * shift_down_pixels
    output[y1+shift_down:y1+shift_down+h, x1:x1+w] = charges_row_roi

    # write final output image
    output_filename = get_filename(timestamp, "output" )
    cv2.imwrite(output_filename, output)
    print(f"[INFO] Output image: {output_filename}")

    ## FR ANALYSIS

    # 06. Read tables with FR layout/document
    print("[INFO] 06. Reading tables with FR layout model...")
    tables = get_document_tables(output_filename)

    ## POSTPROCESSING

    result = {}

    # 07. Navigate and extract data from tables

    # birth date
    result['birth_date'] = f"{tables[0][1]['content']} {tables[0][2]['content']} {tables[0][3]['content']}"

    # items table
    result['items'] = {}
    for cell in tables[1]:
        key = 'row_' + str(cell['row']).zfill(2)
        if cell['row'] in (3, 5, 7, 9, 11, 13):
            if result['items'].get(key) is None: result['items'][key] = {} # fist time needs to initialize the dict             
            if cell['column'] == 0: result['items'][key]['code'] = cell['content']
            elif cell['column'] == 11: result['items'][key]['provider_id'] = cell['content']            
        elif cell['row'] in (4, 6, 8, 10, 12, 14):
            if result['items'].get(key) is None: result['items'][key] = {} # fist time needs to initialize the dict 
            if cell['column'] == 0: result['items'][key]['date_from'] = cell['content']
            elif cell['column'] == 1: result['items'][key]['date_to'] = cell['content']
            elif cell['column'] == 2: result['items'][key]['place_of_service'] = cell['content']
            elif cell['column'] == 3: result['items'][key]['emg'] = cell['content']            
            elif cell['column'] == 4: result['items'][key]['cpt'] = cell['content']
            elif cell['column'] == 5: result['items'][key]['modifier'] = cell['content']            
            elif cell['column'] == 6: result['items'][key]['diagnosis'] = cell['content']
            elif cell['column'] == 7: result['items'][key]['charges'] = cell['content']
            elif cell['column'] == 8: result['items'][key]['units'] = cell['content']
            elif cell['column'] == 11: result['items'][key]['provider_id'] = cell['content']

    # charge table (if exists)
    if len(tables) > 2:
        for cell in tables[2]:
            if cell['row'] == 0:
                if cell['column'] == 0: result['tax_id'] = cell['content']
                elif cell['column'] == 1: result['account_number'] = cell['content']
                elif cell['column'] == 2: result['total_charge'] = cell['content']
                elif cell['column'] == 3: result['amount_paid'] = cell['content']            

    output_text_filename = get_filename(timestamp, "output", extension="json" )
    with open(output_text_filename, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] Output saved to {output_text_filename}")

    print("[INFO] Done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--image",required=True, help="Path to input image that we'll align to template",)
    ap.add_argument("-t", "--template", required=True, help="Path to input template image")
    ap.add_argument("-l", "--labels_mask", required=True, help="Path to labels mask")
    ap.add_argument("-v", "--values_mask", required=True, help="Path to values mask")  
    ap.add_argument("-b", "--tables_frame", required=True, help="Path to tables frame")        
    args = vars(ap.parse_args())
    main(args)
