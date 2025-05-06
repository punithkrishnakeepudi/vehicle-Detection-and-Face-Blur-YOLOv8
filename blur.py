import cv2
from ultralytics import YOLO
import os
import numpy as np

# Ensure YOLOv8 face model is available
if not os.path.exists("yolov8n-face.pt"):
    from ultralytics.utils.downloads import attempt_download_asset
    attempt_download_asset("yolov8n-face.pt")

# Load models
face_model = YOLO("yolov8n-face.pt")   # Face detection
object_model = YOLO("yolov8n.pt")      # Person + bike detection

# Video setup
cap = cv2.VideoCapture("video_2.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Two outputs: one for object detection, one for face blur
out_labels = cv2.VideoWriter("output_yolo_labels.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
out_blur = cv2.VideoWriter("output_blurred_faces.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("🔄 Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    labeled_frame = frame.copy()
    blur_only_frame = frame.copy()

    # Run detections
    obj_results = object_model(frame, verbose=False)[0]
    face_results = face_model(frame, verbose=False)[0]

    bike_boxes = []
    person_boxes = []

    for box in obj_results.boxes:
        cls_id = int(box.cls[0])
        label = object_model.names[cls_id].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in ['motorbike', 'bicycle', 'scooter', 'auto rickshaw', 'motorcycle']:
            bike_boxes.append((x1, y1, x2, y2))
        elif label == 'person':
            person_boxes.append((x1, y1, x2, y2))

    # Add YOLO-style labels to labeled_frame only
    labeled_frame = obj_results.plot()

    # Face blur on associated persons
    for face_box in face_results.boxes:
        fx1, fy1, fx2, fy2 = map(int, face_box.xyxy[0])
        cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2

        for px1, py1, px2, py2 in person_boxes:
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                for bx1, by1, bx2, by2 in bike_boxes:
                    inter_x1 = max(px1, bx1)
                    inter_y1 = max(py1, by1)
                    inter_x2 = min(px2, bx2)
                    inter_y2 = min(py2, by2)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    person_area = (px2 - px1) * (py2 - py1)

                    if inter_area / person_area > 0.3:
                        face = blur_only_frame[fy1:fy2, fx1:fx2]
                        if face.size > 0:
                            blurred = cv2.GaussianBlur(face, (51, 51), 30)
                            blur_only_frame[fy1:fy2, fx1:fx2] = blurred
                            cv2.putText(blur_only_frame, "human head", (fx1, fy1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.rectangle(blur_only_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

    out_labels.write(labeled_frame)
    out_blur.write(blur_only_frame)

cap.release()
out_labels.release()
out_blur.release()
cv2.destroyAllWindows()
print("✅ Done! Videos saved as 'output_yolo_labels.mp4' and 'output_blurred_faces.mp4'")
