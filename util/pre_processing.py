# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import cv2
import numpy as np
import imutils
from util.utils import load_image

# -----------------------------
#   FUNCTIONS
# -----------------------------

def extract_roi(image_path, coord):
    # form rec cood = [x1, y1, x2, y2, x3, y3, x4, y4]
    min_height = 60
    min_width = 60
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x1, y1, x3, y3 = coord[0], coord[1], coord[4], coord[5]
    roi = image[y1:y3, x1:x3]
    height, width = roi.shape
    if height < min_height:
        roi = imutils.resize(roi, height=min_height)
    if width < min_width:
        roi = imutils.resize(roi, width=min_width)
    return roi

def remove_blobs(img, area_threshold):
    # convert image to binary
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
    # remove small blobs
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < area_threshold:
            labels[labels == i] = 0
    # create new image with only large blobs
    new_binary = np.zeros_like(binary)
    new_binary[labels > 0] = 255
    # convert binary image to grayscale
    new_img = cv2.cvtColor(new_binary, cv2.COLOR_GRAY2BGR)
    # invert image
    new_img = cv2.bitwise_not(new_img)
    return new_img

def remove_hlines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # convert image to binary
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    # apply morphological operations
    kernel = np.ones((3, 40), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # apply mask to image
    gray[binary > 0] = 255
    return gray

# -------------------------------------
#   Cropping Charges and Total Charges
# -------------------------------------

def crop_charges(input_image, cv_result):

    # cropping parameters
    roi_max_height = 400
    roi_max_width = 400
    side_border = 7 # pixels
    height_multiplier = 10
    min_charges_detection_confidence = 0.7

    # initialize the cropped images
    cropped_charges = np.zeros((roi_max_width,roi_max_width,3),np.uint8)
    confidence = -1.0
    found_charges = False

    # get highest confidence for charges
    object = {}
    highest_confidence = 0.0
    for obj in cv_result['customModelResult']['objectsResult']['values']:
        for tag in obj['tags']:
            if tag['confidence'] > highest_confidence and tag['name'] == 'charges':
                highest_confidence = tag['confidence']
                object = obj

   # check if object was found
    if object != {}:
        for tag in object['tags']:
            if tag['confidence'] >= min_charges_detection_confidence and tag['name'] == 'charges':
                image = cv2.imread(input_image)                
                x = object['boundingBox']['x']
                y = object['boundingBox']['y']
                w = object['boundingBox']['w']
                h = object['boundingBox']['h']

                roi_height = height_multiplier*h
                roi_height = min(roi_height, roi_max_height)
                cropped_charges = image[y+h:y+h+roi_height, x+side_border:x+w-side_border]
              
                confidence = tag['confidence']
                found_charges = True

    return cropped_charges, confidence, found_charges

def crop_total_charges(input_image, cv_result):

    # cropping parameters
    roi_max_height = 400
    roi_max_width = 400
    side_border = 1 # pixels
    height_multiplier = 4.5
    width_multiplier= 1.3
    min_total_charges_detection_confidence = 0.20

    # initialize the cropped images
    cropped_charges = np.zeros((roi_max_height,roi_max_width,3),np.uint8)
    cropped_charges[:,:] = (255,255,255)
    confidence = -1.0
    found_charges = False

    # get highest confidence for total charges
    object = {}
    highest_confidence = 0.0
    for obj in cv_result['customModelResult']['objectsResult']['values']:
        for tag in obj['tags']:
            if tag['confidence'] > highest_confidence and tag['name'] == 'totacharges':
                highest_confidence = tag['confidence']
                object = obj
    
    # check if object was found
    if object != {}:
        for tag in object['tags']:
            if tag['confidence'] >= min_total_charges_detection_confidence and tag['name'] == 'totacharges':
                image = cv2.imread(input_image)
                x = object['boundingBox']['x']
                y = object['boundingBox']['y']
                w = object['boundingBox']['w']
                h = object['boundingBox']['h']

                roi_height = int(height_multiplier*h)
                roi_width = int(width_multiplier*w)
                
                crop = image[y+h:y+roi_height, x+side_border:x+roi_width]
                cropped_charges[:crop.shape[0], :crop.shape[1]] = crop
                
                confidence = tag['confidence']
                found_charges = True

    return cropped_charges, confidence, found_charges

def crop_dates(input_image, cv_result):

    # cropping parameters
    roi_max_height = 390
    roi_max_width = 400
    side_border = 7 # pixels
    height_multiplier = 10
    min_dates_detection_confidence = 0.7

    # initialize the cropped images
    cropped_dates = np.zeros((roi_max_height,roi_max_width,3),np.uint8)
    confidence = -1.0
    found_charges = False

    # get highest confidence for charges
    object = {}
    highest_confidence = 0.0
    for obj in cv_result['customModelResult']['objectsResult']['values']:
        for tag in obj['tags']:
            if tag['confidence'] > highest_confidence and tag['name'] == 'datesofservice':
                highest_confidence = tag['confidence']
                object = obj

   # check if object was found
    if object != {}:
        for tag in object['tags']:
            if tag['confidence'] >= min_dates_detection_confidence and tag['name'] == 'datesofservice':
                image = cv2.imread(input_image)                
                x = object['boundingBox']['x']
                y = object['boundingBox']['y']
                w = object['boundingBox']['w']
                h = object['boundingBox']['h']

                roi_height = height_multiplier*h
                roi_height = min(roi_height, roi_max_height)
                cropped_dates = image[y+h:y+h+roi_height, x+side_border:x+w-side_border]
              
                confidence = tag['confidence']
                found_charges = True

    return cropped_dates, confidence, found_charges

# -----------------------------
#   CURRENTLY NOT IN USE
# -----------------------------

def shift_down(image, shift=1, highlight=[0,0,255]):
    height, width, channels = image.shape
    shifted_image = image.copy()
    for i in range(height):
        for j in range(width):
            if (image[i,j] == highlight).all():
                if i < height-shift:
                    shifted_image[i,j] = [255,255,255]
    for i in range(height):
        for j in range(width):
            if (image[i,j] == highlight).all():
                if i < height-shift:
                    shifted_image[i+shift,j] = highlight                    
    return shifted_image

def shift_values(image, template, shift=1, highlight=[0,0,255]):
    # 01. transform the aligned image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 02. create a mask based on the template (form with empty fields)
    # mask will be used to remove template elements from the image
    mask = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    threshold = 250
    assign_value = 255
    threshold_method = cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(mask,threshold,assign_value,threshold_method)

    # debug_file = 'data/debug.jpg'
    # cv2.imwrite(debug_file, image)

    # 03. dilates mask to remove small differences
    kernel = np.ones((3,2),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    # 04. remove template elements from image based on mask
    masked_image = gray.copy()
    template_elements = mask > 0
    masked_image[template_elements] = 255

    # 05. highlight the fields that are not empty 
    gray_threshold = 160
    threshold_method = cv2.THRESH_BINARY
    _, highlight_mask = cv2.threshold(masked_image,gray_threshold,assign_value,threshold_method)
    highlight_mask = cv2.cvtColor(highlight_mask, cv2.COLOR_GRAY2BGR)
    highlight_image= np.where(highlight_mask == [0,0,0], [0,0,255], image)
 
    # 06. highlight the template text 
    # gray_threshold = 125
    # threshold_method = cv2.THRESH_BINARY
    # _, highlight_mask = cv2.threshold(masked_image,gray_threshold,assign_value,threshold_method)
    # template_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # highlight_image= np.where(template_mask == [255,255,255], [255,255,255], highlight_image)

    highlight_image = shift_down(highlight_image, shift=5, highlight=[0,0,255])
    highlight_image = cv2.convertScaleAbs(highlight_image)

    # 08. return the image without the background
    return highlight_image

def is_not_inside_bbox(contour,words):
    for box in words:
        x,y,w,h = box
        for point in contour:
            if x <= point[0][0] and y <= point[0][1] and x+w >= point[0][0] and y+h >= point[0][1]:
                return False
    return True

def remove_vertical_lines_old(image, words):
    result = image.copy()
    result_debug = image.copy()    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        if is_not_inside_bbox(c,words):
            cv2.drawContours(result_debug, [c], -1, (36,255,12), 2)
            cv2.drawContours(result, [c], -1, (255,255,255), 2)
    return result, result_debug

def fill_small_holes(img):
    # Convert image to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img.copy()

    # Threshold the image to binary
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Create a kernel for morphological operations
    kernel = np.ones((2,3), np.uint8)

    # Perform morphological closing to fill small holes inside letters
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Invert the image back to its original form
    filled = cv2.bitwise_not(closing)

    return filled

def remove_template(gray, template):
    # 01. create a mask based on the template (form with empty fields)
    # mask will be used to remove template elements from the image
    mask = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    threshold = 200
    assign_value = 255
    threshold_method = cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(mask,threshold,assign_value,threshold_method)

    # debug_file = 'data/debug.jpg'
    # cv2.imwrite(debug_file, image)

    # 02. dilates mask to remove small differences
    kernel = np.ones((2,2),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    # 03. remove template elements from image based on mask
    masked_image = gray.copy()
    template_elements = mask > 0
    masked_image[template_elements] = 255

    # masked_image = fill_small_holes(masked_image) # Experimental

    return masked_image