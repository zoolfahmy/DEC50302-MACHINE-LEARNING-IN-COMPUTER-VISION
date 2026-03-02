import cv2
from pyzbar.pyzbar import decode

# Requires OpenCV build with barcode support (often easiest via opencv-contrib-python)
bardet = cv2.barcode.BarcodeDetector()

img = cv2.imread("BarcodeEAN-13.png")
if img is None:
    raise FileNotFoundError("Provide barcode.png")

ok, decoded_info, decoded_type, corners = bardet.detectAndDecodeWithType(img)

print("OK:", ok)
print("Decoded:", decoded_info)
print("Types:", decoded_type)

# Visualize detections if any
if corners is not None and len(corners) > 0:
    for quad in corners:
        quad = quad.astype(int).reshape(-1, 2)
        for i in range(4):
            cv2.line(img, tuple(quad[i]), tuple(quad[(i + 1) % 4]), (0, 255, 0), 2)

cv2.imwrite("barcode_detected.png", img)