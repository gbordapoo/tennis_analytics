class BallDetector:
    def __init__(self, model_path: str) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        balls = []
        for r in results:
            for box in r.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                balls.append(bbox)
        return balls
