# local_detector.py

import cv2
import os
from datetime import datetime
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can replace with 'yolov8s.pt' or your trained model

# Set your camera stream (IP or local)
CAMERA_URL = 'http://100.94.103.96:8080/video'
CAMERA_NAME = 'classroom_A'

# Setup folders
screenshot_dir = 'screenshots'
os.makedirs(screenshot_dir, exist_ok=True)

log_file = 'detection_log.csv'
if not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write('Timestamp,Camera\n')

# Start camera
cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print(f"[ERROR] Cannot open camera: {CAMERA_URL}")
    exit()

print("[INFO] Camera stream started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    # Run detection
    results = model(frame)

    detected = False
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            cls_name = r.names[class_id]
            if cls_name == 'cell phone':
                detected = True
                # Save screenshot
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{CAMERA_NAME}_{timestamp}.jpg"
                cv2.imwrite(os.path.join(screenshot_dir, filename), frame)

                # Log detection
                with open(log_file, 'a') as f:
                    f.write(f"{timestamp},{CAMERA_NAME}\n")

                print(f"[DETECTED] Mobile phone at {timestamp}")

    # (Optional) Show the frame in a window
    cv2.imshow("Live Detection", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
