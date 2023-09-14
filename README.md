# 1500 Form Insurance Claim data Extraction

Health Insurance 1500 Claim Forms preprocessing and postprocessing scripts for Form Recognizer.

### Scripts

- process_1500.py: extracts charges and total charges.

### Pre-reqs

- Poppler: ```Conda install -c conda-forge poppler```

- Object detection model trained to detect **charges**, **totacharges**, **datesofservice**, **qty** and **birthdate** objects.

### How to run?

```python process_1500.py -i [FOLDER_WHERE_PDFS_ARE_OR_SINGLE_PDF_PATH]```


