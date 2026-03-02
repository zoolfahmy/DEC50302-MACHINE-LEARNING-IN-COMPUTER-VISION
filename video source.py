import cv2

cap = cv2.VideoCapture(0)  # 0 = default camera; for a file: cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    raise RuntimeError("Could not open camera/video source.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Original video", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale video", gray)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()