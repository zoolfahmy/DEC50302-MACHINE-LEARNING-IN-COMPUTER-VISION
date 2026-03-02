import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("digit_model.keras")

img = cv2.imread("digit4.png")
if img is None:
    raise FileNotFoundError("Provide a digit.png image with one digit to classify.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Normalize lighting and binarize (try both THRESH_BINARY and THRESH_BINARY_INV depending on background)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Optional cleanup: remove small noise, close small gaps
kernel = np.ones((3, 3), np.uint8)
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

# Resize to MNIST size and normalize
roi = cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)
roi = roi.astype("float32") / 255.0
roi = roi[None, ..., None]  # (1, 28, 28, 1)

# Add this before resizing to see what the model actually sees
plt.figure(figsize=(4, 4))
plt.imshow(roi[0, :, :, 0], cmap='gray')
plt.title("What the model sees")
plt.show()

probs = model.predict(roi, verbose=0)[0]
pred = int(np.argmax(probs))

print("Prediction:", pred)
print("Probabilities:", probs.round(3))