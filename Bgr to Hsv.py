import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read or load image from its path
img = cv2.imread(r"D:/Machine Learning Course/SA github/chapter2/images/nature.jpg")
if img is None:
    raise FileNotFoundError("Provide nature.jpg (or check working directory).")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale image", gray)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

plt.figure()
plt.title("Grayscale intensity histogram")
plt.xlabel("Pixel intensity (0-255)")
plt.ylabel("Count")
plt.plot(hist)
plt.show()