import cv2
import numpy as np
from tensorflow.keras.models import load_model  


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# 1. Load the model and the labels
print("Loading model...")
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
print("Model loaded successfully!")

# 2. Open the default webcam (Camera 0)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the program.")

while True:
    # 3. Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Preprocess the image to match the format Teachable Machine expects
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)
    normalized_image_array = (image_array / 127.5) - 1

    # 5. Run the prediction
    prediction = model.predict(normalized_image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # 6. Display the result on the screen
    text = f"{class_name}: {np.round(confidence_score * 100)}%"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Teachable Machine Local Inference", frame)

    # 7. Listen for the 'q' key to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
