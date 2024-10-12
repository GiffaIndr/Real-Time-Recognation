import cv2
import mediapipe as mp
import numpy as np
import os


prototxt_path = 'C:\\My Stuff\\pitong\\MobileNetSSD_deploy.prototxt'
model_path = 'C:\\My Stuff\\pitong\\MobileNetSSD_deploy.caffemodel'

print("Prototxt exists:", os.path.exists(prototxt_path))
print("Caffemodel exists:", os.path.exists(model_path))

try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("Model loaded successfully.")
except cv2.error as e:
    print(f"Error loading model: {e}")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

face_detection = mp_face_detection.FaceDetection()
hands = mp_hands.Hands()

class_names = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_detection.process(rgb_frame)

    hand_results = hands.process(rgb_frame)

    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)

            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)

            cv2.putText(frame, "orang ganteng", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
            print("Wajah terdeteksi")

    else:
        print("Wajah tidak terdeteksi")

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{class_names[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Mediapipe Face, Hand Detection & Object Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
