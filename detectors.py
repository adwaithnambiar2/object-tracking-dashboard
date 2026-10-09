from pathlib import Path
import cv2

CLASSES = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7, "cat": 15, "dog": 16}

class YoloDetector:
    """Use pretrained YOLO11n for inference only. No model training occurs here."""
    def __init__(self, labels, confidence=0.35):
        from ultralytics import YOLO
        weights = Path(__file__).resolve().parent / "weights"
        weights.mkdir(exist_ok=True)
        self.model = YOLO(str(weights / "yolo11n.pt"))
        self.class_ids = [CLASSES[name] for name in labels]
        self.confidence = confidence

    def detect(self, frame, frame_index=0):
        result = self.model.predict(frame, classes=self.class_ids, conf=self.confidence,
                                    imgsz=640, device="cpu", verbose=False)[0]
        boxes = result.boxes
        if boxes is None:
            return []
        return [{"box": [float(v) for v in box], "label": self.model.names[int(cls)], "confidence": float(conf)}
                for box, cls, conf in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist())]

class DemoDetector:
    """Threshold known colored shapes. This is NOT a learned object detector."""
    def detect(self, frame, frame_index=0):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []
        for label, low, high in [("demo_green", (40, 120, 100), (85, 255, 255)),
                                  ("demo_blue", (95, 120, 100), (130, 255, 255))]:
            mask = cv2.inRange(hsv, low, high)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 200:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({"box": [x, y, x+w, y+h], "label": label, "confidence": 1.0})
        return detections
