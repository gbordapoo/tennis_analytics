from ultralytics import YOLO


class PlayerDetector:
    def __init__(self, model: str = "yolov8n.pt") -> None:
        self.model = YOLO(model)

    def detect(self, frame):
        results = self.model(frame)
        players = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # person class
                    bbox = box.xyxy[0].cpu().numpy()
                    players.append(bbox)
        return players
