from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


class BallDetector:
    def __init__(self, model_path: str, conf: float = 0.2) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Ball model not found: {model_file}")
        self.model = YOLO(str(model_file))
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        balls = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                balls.append((float(x1), float(y1), float(x2), float(y2)))
        return balls
