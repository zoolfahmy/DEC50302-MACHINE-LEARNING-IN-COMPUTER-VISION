import cv2
import json
import mediapipe as mp

print("Loading metadata...")
with open("metadata.json", "r") as f:
    metadata = json.load(f)

print("Classes:", metadata["labels"])

print("Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Webcam not found!")
    exit()

print("Webcam opened!")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print("Loading MediaPipe Pose...")
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)

def detect_hands_up(landmarks):
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    return left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y

print("Ready! Press 'q' to quit.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam")
        break

    frame = cv2.flip(frame, 1)

    # Tukar ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        results = pose.process(rgb_frame)
    except Exception as e:
        print("MediaPipe error:", e)
        break

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        if detect_hands_up(landmarks):
            cv2.putText(frame, "HANDS UP DETECTED!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Hands Down", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No pose detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, f"Frame: {frame_count}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Pose Detection", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()
print("Done!")