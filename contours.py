import cv2
import numpy as np

img = cv2.imread("shapes_contours.png")
if img is None:
    raise FileNotFoundError("Run the previous contour script first to create shapes_contours.png.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # low threshold since background is black-ish

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

out = img.copy()

for c in contours:
    peri = cv2.arcLength(c, True)
    epsilon = 0.02 * peri  # 2% of perimeter is a common starting point
    approx = cv2.approxPolyDP(c, epsilon, True)

    x, y, w, h = cv2.boundingRect(approx)
    vertices = len(approx)

    if vertices == 3:
        label = "triangle"
    elif vertices == 4:
        label = "quadrilateral"
    else:
        label = f"{vertices}-gon/circle-ish"

    cv2.drawContours(out, [approx], -1, (0, 0, 255), 2)
    cv2.putText(out, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

cv2.imwrite("shapes_approximated.png", out)
cv2.imshow("Approximated contours", out)
cv2.waitKey(0)  # keep window open until a key press    