# Insurance Claim Preprocessing

### General Experimentation Notes

**Problems:**

P1) Labels connected or with little overlapping to the values. Ex: PATIENT'S BIRTH DATE = 03! 20 ! 55"

P2) Vertical bars: Ex1: Charges = 682 |00   Ex2: Place of Service = |10

P3) Documents with high overlapping between labels and values.


**Hypotheses:**

H1) Improve P1 and P2 by aligning document to nucc template to separate values from lables and remove vertical lines.

H1B) Improve P1 and P2 by aligning document to nucc template to differenciate template elements from values by changing template's pixel value.

H2) Improve P1 and P2 by aligning document to nucc template then keep only values by applying values masks and removing template elements.

H3) Improve P1 and P2 by aligning document to nucc template then keep elements in a table format by applying fields masks, removing template elements then adding table frame.


**General Comments:**

C1) Alignemnt is good but is not working in all cases.

C2) Need to investigate further the impact of the 'keep percentage' parameter on the results.