from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


@dataclass
class VideoInfo:
    fps: float
    frame_width: int
    frame_height: int
    total_frames: int


def load_model(model_path: Path) -> YOLO:
    return YOLO(str(model_path))


def _sports_ball_class_id(model: YOLO) -> int | None:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        for cid, name in names.items():
            if str(name).strip().lower() == "sports ball":
                return int(cid)
    return None


def run_detection(model: YOLO, video_path: Path, conf: float = 0.15) -> tuple[list, pd.DataFrame, VideoInfo]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_info = VideoInfo(float(fps), frame_width, frame_height, total_frames)
    frames_raw = []
    detecciones = []
    sports_ball_id = _sports_ball_class_id(model)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frames_raw.append(frame.copy())

        results = model.predict(frame, conf=conf, verbose=False)
        best_box = None
        best_conf = -1.0

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                if sports_ball_id is not None and cls_id != sports_ball_id:
                    continue
                score = float(box.conf[0]) if box.conf is not None else 0.0
                if score > best_conf:
                    best_conf = score
                    best_box = box

        if best_box is not None:
            x1, y1, x2, y2 = best_box.xyxy[0].tolist()
            detecciones.append(
                {
                    "frame": frame_count,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": round(float(best_conf), 4),
                }
            )

    cap.release()

    df_detecciones = pd.DataFrame(detecciones)
    if not df_detecciones.empty:
        df_detecciones = df_detecciones.sort_values("frame").reset_index(drop=True)
    return frames_raw, df_detecciones, video_info
