# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import cv2
import imutils
import os
import shutil
from pdf2image import convert_from_path

# -----------------------------
#   FUNCTIONS
# -----------------------------

# set working directory
WORK_DIR = 'work'

def write_image(filename, image, width="", dpi=200):
    if width == "":
        output_image = cv2.resize(image, None, fx=float(dpi)/72, fy=float(dpi)/72, interpolation=cv2.INTER_CUBIC)
    else:
        output_image = image.copy()
        imutils.resize(output_image, width=width)

    cv2.imwrite(filename, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

def write_debug_image(image, width):
    write_image('work/debug.jpg', image, width)

def get_work_filename(prefix, sufix, filename="", extension="jpg"):
    """
    This function takes in a filename, a prefix and a suffix and returns a path to a working file.
    If the filename exists it will be moved to the working directory.
    If the filename is a pdf and extension is jpg, it will be converted to a jpg file.
    """    
    if not os.path.exists(WORK_DIR): os.makedirs(WORK_DIR)
    work_file = f"{WORK_DIR}/{prefix}-{sufix}.{extension}"
    if extension == "jpg" and filename.endswith(".pdf"):
        # needs to convert pdf to jpg
        pages = convert_from_path(filename, dpi=200)
        pages[0].save(work_file, 'JPEG')
    elif os.path.exists(filename):
        shutil.copy(filename, work_file)
    return work_file
