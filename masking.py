import cv2
import numpy as np

img = cv2.imread("D:/Machine Learning Course/SA github/chapter2/images/green tree.jpg")
if img is None:
    raise FileNotFoundError("Provide green tree.jpg (or check working directory).")
cv2.imshow("Original image", img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Hue ranges depend on lighting/camera. These are starter values for "green".
lower_green = np.array([35, 80, 80], dtype=np.uint8)
upper_green = np.array([85, 255, 255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_green, upper_green)

# Apply the mask: keep only pixels where mask != 0
segmented = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite("mask.png", mask)
cv2.imshow("Mask", mask)
cv2.imwrite("segmented_green.png", segmented)
cv2.imshow("Segmented green areas", segmented)  
cv2.waitKey(0)  # keep window open until a key press                        
cv2.destroyAllWindows()
