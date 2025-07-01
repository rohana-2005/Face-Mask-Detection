import cv2
import numpy as np
import joblib

# Same order as training!
categories = ['with_mask', 'without_mask']

# Load model and scaler
model = joblib.load("face_mask_model.pkl")
scaler = joblib.load("scaler.pkl")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    img = cv2.resize(frame, (64, 64)).flatten().reshape(1, -1)
    img = scaler.transform(img)

    pred = model.predict(img)[0]
    label = categories[pred]

    color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)

    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
