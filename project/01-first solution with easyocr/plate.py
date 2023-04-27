import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

plate_img = cv2.imread('plate2.jpg')
gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
mask = np.zeros(gray.shape, np.uint8)

bilateral = cv2.bilateralFilter(gray, 11, 15, 30)  # bilateralFilter: Smoothing and Blurring Images
edge = cv2.Canny(bilateral, 90, 150)
contours = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
refined_contours = imutils.grab_contours(contours)  # The obtained counters are congregated
sorted_contours = sorted(refined_contours, key=cv2.contourArea, reverse=True)[:4]

for contour in sorted_contours:
    approx = cv2.approxPolyDP(contour, 10, True)  # approxPolyDP() is use sides closed lines for approximate estimation
    if len(approx) == 4:
        plate_length = approx
        print(plate_length)
        break

drawing = cv2.drawContours(mask, [plate_length], 0, 255, -1)
final = cv2.bitwise_and(plate_img, plate_img, mask=drawing)

plt.imshow(cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
plt.show()
