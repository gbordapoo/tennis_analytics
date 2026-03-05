from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


class PlayerDetector:
    def __init__(self, model_path: str, conf: float = 0.25) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Player model not found at {model_file}. Please download a local YOLOv8 weights file manually "
                "(e.g., models/yolov8n.pt) and pass --player-model."
            )
        self.model = YOLO(str(model_file))
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        players = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                players.append((float(x1), float(y1), float(x2), float(y2)))
        return players
