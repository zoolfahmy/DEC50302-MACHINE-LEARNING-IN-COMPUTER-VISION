import cv2
import numpy as np

def make_demo_scene(width=640, height=400):
    """
    Create a synthetic BGR image so you can practice without downloading files.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # dark gray background (BGR)

    # Draw colored shapes (BGR order!)
    cv2.rectangle(img, (30, 40), (220, 200), (0, 0, 255), -1)   # red rectangle
    cv2.circle(img, (340, 120), 70, (0, 255, 0), -1)            # green circle
    pts = np.array([[470, 220], [600, 220], [535, 320]], np.int32)
    cv2.fillPoly(img, [pts], (255, 0, 0))                       # blue triangle

    cv2.putText(img, "OpenCV demo", (30, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img

img = make_demo_scene()
cv2.imwrite("demo.png", img)          # save (BGR PNG)
loaded = cv2.imread("demo.png")       # read back

if loaded is None:
    raise FileNotFoundError("Failed to read demo.png (check working directory).")

cv2.imshow("Loaded image", loaded)
cv2.waitKey(0)                        # keep window open until a key press
cv2.destroyAllWindows()