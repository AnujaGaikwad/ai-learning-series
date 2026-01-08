import cv2
import random
import time
from collections import deque

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cam = cv2.VideoCapture(0)

# Emotion labels (demo)
emotions = ["Happy", "Sad", "Angry", "Surprised", "Neutral"]

# Store recent emotion predictions
emotion_history = deque(maxlen=15)

current_emotion = "Neutral"
last_update_time = time.time()
UPDATE_INTERVAL = 2  # seconds

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Resize frame for stability
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Simulate emotion prediction (demo)
        predicted_emotion = random.choice(emotions)
        emotion_history.append(predicted_emotion)

        # Update emotion slowly (temporal smoothing)
        if time.time() - last_update_time > UPDATE_INTERVAL:
            current_emotion = max(
                set(emotion_history),
                key=emotion_history.count
            )
            last_update_time = time.time()

        # Display emotion
        cv2.putText(
            frame,
            current_emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    cv2.imshow("Face Emotion Recognition (Stable Demo)", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
