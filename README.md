# Insurance Claim Preprocessing

Preprocessing experimentations with Health Insurance 1500 Claim Forms template to improve data extraction using Form Recognizer.

**process_1500.py**

General processing flow steps

- Detect QR Code
- Crop image based on QR Code position
- For each field
    - Apply field's mask
    - Remove background noise (blobs) *
    - Analyze document with FR layout (high resolution on)
    - Extract values with regex
    - Use LLM to infer a valid value if applicable *

\* not used in some fields