# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import cv2
import imutils
import os
import shutil
from pdf2image import convert_from_path
import logging

# -----------------------------
#   FUNCTIONS
# -----------------------------

# set working directory
WORK_DIR = 'work'

def load_image(image_filename, timestamp, prefix, gray=False, width=-1):
    # define working files
    work_image = get_filename(timestamp, prefix, image_filename)
    # Load the input image and template
    image = cv2.imread(work_image)
    if gray: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the images
    if width > 0:
    # print(f"[INFO] Loaded and resized width to {width}: {image_filename}")
        image = imutils.resize(image, width=width)
    return image

def write_debug_image(image, width=1700):
    output_image = image.copy()
    imutils.resize(output_image, width=width)
    cv2.imwrite(f'work/debug.jpg', output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])    

def get_filename(prefix, sufix, filename="", extension="jpg"):
    """
    This function takes in a filename, a prefix and a suffix and returns a path to a working file.
    If the filename exists it will be moved to the working directory.
    If the filename is a pdf and extension is jpg, it will be converted to a jpg file.
    """
    work_dir = WORK_DIR
    if not os.path.exists(work_dir): os.makedirs(work_dir)
    work_file = f"{work_dir}/{prefix}-{sufix}.{extension}"
    if extension == "jpg" and filename.endswith(".pdf"):
        # needs to convert pdf to jpg
        try:
            pages = convert_from_path(filename, dpi=200)
            pages[0].save(work_file, 'JPEG')
        except Exception as e:
            logging.error(f"Could not get the first page, check if poppler is installed. {e}")
            exit()        

    elif os.path.exists(filename):
        shutil.copy(filename, work_file)
    return work_file

def count_digits(s):
    count = 0
    for char in s:
        if char.isdigit():
            count += 1
    return count

