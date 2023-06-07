# Insurance Claim Preprocessing

Preprocessing experimentations with Health Insurance Claim Forms template to improve data extraction using Form Recognizer.

[General Notes](./Notes.md)

**process.py**

- Align document to template
- Apply masks to keep only the field labels and values
- Remove template's background from final image
- Add table frame
- Run FR layout or custom model analysis

**References**

[Image alignment and registration with OpenCV](https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/)