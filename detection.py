import cv2
import os
import time
from ultralytics import YOLO
from datetime import datetime
from helpers.logger import log_detection
from helpers.pause_manager import is_paused

# Load the YOLOv8 model (use yolov8n.pt for Render memory limits)
model = YOLO("yolov8n.pt")

# Cooldown tracker dictionary
cooldown_tracker = {}  # {camera_name: last_capture_time}

# Cooldown duration in seconds (15s)
COOLDOWN_TIME = 15

def generate_frames(camera_url, camera_name):
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {camera_name}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Pause detection if user requested
        if is_paused():
            continue

        # Run detection
        results = model.predict(frame, conf=0.5)
        detected = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                if class_name == 'cell phone':
                    detected = True
                    break

        # Screenshot saving with cooldown
        if detected:
            now = time.time()
            last_capture = cooldown_tracker.get(camera_name, 0)

            if now - last_capture >= COOLDOWN_TIME:
                save_screenshot(frame, camera_name)
                cooldown_tracker[camera_name] = now
                log_detection(camera_name)
            else:
                print(f"[{camera_name}] Cooldown active. Skipping screenshot.")

        # Encode frame for web streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def save_screenshot(frame, camera_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{camera_name}_{timestamp}.jpg"
    folder = os.path.join("static", "screenshots")

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    cv2.imwrite(path, frame)
    print(f"[INFO] Saved screenshot: {path}")
