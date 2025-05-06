🚘 Vehicle Detection and Face Blur using YOLOv8

This project detects vehicles and people using YOLOv8 object detection, and automatically blurs the faces of riders on two-wheelers or scooters. It produces two separate output videos:

1. YOLO Labels Output: Visualizes object detection with bounding boxes (YOLO-style).
2. Face Blur Output: Detects and blurs faces of riders and labels them as "human head".

---

📌 Key Features

- Object detection using YOLOv8
- Face detection using YOLOv8n-face
- Automatic blurring of faces of persons riding bikes or scooters
- Saves two processed video outputs
- Clean and modular Python code

---

🛠️ Tech Stack

| Component       | Description                        |
|----------------|------------------------------------|
| Python          | Programming language used         |
| OpenCV          | Image processing and video handling |
| Ultralytics YOLOv8 | Object and face detection     |
| NumPy           | Matrix and pixel manipulation      |

---

🎯 Detection Targets

| Object Type                                    | Action                       |
|------------------------------------------------|------------------------------|
| motorbike, scooter, bicycle, auto rickshaw, motorcycle | Detected and tagged     |
| person                                         | Detected and linked with vehicle |
| Faces of linked persons                        | Blurred and labeled          |
