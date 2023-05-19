# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import cv2
import numpy as np
import math

# -----------------------------
#   FUNCTIONS
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

    # 08. return the image without the background
    return highlight_image

def is_not_inside_bbox(contour,words):
    for box in words:
        x,y,w,h = box
        for point in contour:
            if x <= point[0][0] and y <= point[0][1] and x+w >= point[0][0] and y+h >= point[0][1]:
                return False
    return True


def remove_vertical_lines(image, words):
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
