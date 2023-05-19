# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import imutils
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------
def align(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    # Convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # Use ORB to detect keypoints in the image and extract (binary) local invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # Match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    # Sort out the matches by their distance (the smaller the distance, the 'more similar' the features are)
    matches = sorted(matches, key=lambda x: x.distance)
    # Keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # Check to see if the matched keypoints should be visualized
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)
    # Allocate the memory for the keypoints (x, y coordinates) from the top matches
    # These coordinates are going to be used to compute the homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # Loop over the top matches
    for (i, m) in enumerate(matches):
        # Indicate that the two keypoints in the respective images map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt
    # Compute the homography matrix between the two sets of matched points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # Use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    # Return the aligned image
    return aligned