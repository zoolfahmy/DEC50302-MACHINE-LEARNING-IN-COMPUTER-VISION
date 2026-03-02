import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("D:/Machine Learning Course/SA github/chapter2/images/soccer-in-green.jpg")
cv2.imshow("Original image", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def edge_count(t1, t2):
    edges = cv2.Canny(gray, t1, t2)
    return int((edges > 0).sum()), edges

# Save and show one example edge map
count, edges = edge_count(50, 150)
cv2.imwrite("edges_50_150.png", edges)
cv2.imshow("Canny edges (t1=50, t2=150)", edges)
print(f"Edge pixels detected with t1=50, t2=150: {count}")

# Sweep thresholds and plot edge counts
t1_values = list(range(10, 200, 10))
counts = []
for t1 in t1_values:
    t2 = min(255, 3 * t1)   # common heuristic: high threshold ~ 2-3x low threshold
    c, _ = edge_count(t1, t2)
    counts.append(c)

plt.figure()
plt.title("Canny sensitivity curve (soccer-in-green.jpg)")
plt.xlabel("Lower threshold (t1)")
plt.ylabel("Edge pixels detected")
plt.plot(t1_values, counts)
plt.show()