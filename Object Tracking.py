import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

lower_green = np.array([35, 80, 80], dtype=np.uint8)
upper_green = np.array([85, 255, 255], dtype=np.uint8)

kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up mask: opening removes small noise; closing fills small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        # Ignore tiny detections
        if area > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"area={int(area)}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Tracking (press q to quit)", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()