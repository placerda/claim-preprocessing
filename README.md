# Insurance Claim Preprocessing

Health Insurance Claim Forms preprocessing to extract form values with Form Recognizer.

Two approaches:

**preprocess_01.py**

- Align document to template [Reference](https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/)
- Shift values down 
- Remove vertical lines (needs improvement)
- Run FormRec document analysis

**preprocess_02.py**

- Align document to template [Reference](https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/)
- Apply masks to keep only the labels and values in the document
- Run FormRec document analysis

**TODO**

TODO 1: Test and work on the alignment dependency on input width.
TODO 2: Improve separators removal.
