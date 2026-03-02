import cv2
import numpy as np

# Create a clean shapes-only image (white shapes on black) for predictable contours
canvas = np.zeros((400, 640, 3), dtype=np.uint8)
cv2.rectangle(canvas, (50, 50), (220, 220), (255, 255, 255), -1)
cv2.circle(canvas, (350, 140), 80, (255, 255, 255), -1)
pts = np.array([[480, 250], [600, 250], [540, 350]], np.int32)
cv2.fillPoly(canvas, [pts], (255, 255, 255))

gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

out = canvas.copy()
cv2.drawContours(out, contours, -1, (0, 255, 0), 3)

cv2.imwrite("shapes_binary.png", binary)
cv2.imshow("Binary image", binary)
cv2.imwrite("shapes_contours.png", out)
cv2.imshow("Contours drawn", out)

print("Contours found:", len(contours))
for i, c in enumerate(contours):
    print(f"Contour {i}: {len(c)} points, area={cv2.contourArea(c)}")
cv2.waitKey(0)  # keep windows open until a key press
cv2.destroyAllWindows()