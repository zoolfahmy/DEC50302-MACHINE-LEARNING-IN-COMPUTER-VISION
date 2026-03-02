import cv2

# Read or load image from its path
img = cv2.imread(r"D:/Machine Learning Course/SA github/chapter2/images/nature.jpg")
if img is None:
    raise FileNotFoundError("Provide nature.jpg (or check working directory).")

# Display the image and wait until a key is pressed
cv2.imshow("Original image", img)
# Convert to grayscale (single channel)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("demo_gray.png", gray)
print("Gray shape:", gray.shape)  # (H, W)
cv2.imshow("Grayscale image", gray)
cv2.waitKey(0)                      # keep window open until a key press                        
cv2.destroyAllWindows()
