import cv2
import imutils
import os
from glob import glob
from pdf2image import convert_from_path

PDF_DIR = 'data/inputs/pdf'
OUTPUT_DIR = 'data/outputs'
WIDTH = 1700
RESIZE = True

''' this function converts all pdfs in a directory to jpegs '''
def convert_pdfs_to_jpegs():
    pdfs = glob(f"{PDF_DIR}/*.pdf")
    if not os.path.exists(OUTPUT_DIR):  os.makedirs(OUTPUT_DIR)
    for pdf in pdfs:
        print(f"[INFO] Converting {pdf} to jpeg...")
        pages = convert_from_path(pdf, dpi=200)
        name = pdf.split("/")[-1].split(".")[0]
        output = f"{OUTPUT_DIR}/{name}.jpg"
        pages[0].save(output, 'JPEG')
        if RESIZE:
            jpeg_image = cv2.imread(output)
            jpeg_image = imutils.resize(jpeg_image, width=WIDTH)
            cv2.imwrite(output, jpeg_image)

print(f"[INFO] Starting script.")
convert_pdfs_to_jpegs()