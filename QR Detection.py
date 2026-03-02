import cv2

detector = cv2.QRCodeDetector()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    text, points, _ = detector.detectAndDecode(frame)

    if points is not None and len(points) > 0:
        pts = points.astype(int).reshape(-1, 2)
        # draw polygon
        for i in range(len(pts)):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)

    if text:
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("QR scanner (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()