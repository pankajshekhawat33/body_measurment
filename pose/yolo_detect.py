import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("models/yolov8n.pt")


def detect_person(image_path):
    """
    Detect person and return cropped body image
    """

    img = cv2.imread(image_path)

    results = model(img)

    boxes = results[0].boxes

    best_box = None

    for box in boxes:

        cls = int(box.cls[0])

        # YOLO class 0 = person
        if cls == 0:
            best_box = box
            break

    if best_box is None:
        return None

    x1, y1, x2, y2 = map(
        int,
        best_box.xyxy[0]
    )

    crop = img[y1:y2, x1:x2]

    return crop