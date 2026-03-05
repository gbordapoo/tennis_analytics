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


def run_detection(model: YOLO, video_path: Path) -> tuple[list, pd.DataFrame, VideoInfo]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # fallback

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_info = VideoInfo(
        fps=float(fps),
        frame_width=frame_width,
        frame_height=frame_height,
        total_frames=total_frames,
    )

    frames_raw = []
    detecciones = []

    # Primera pasada: detección
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frames_raw.append(frame.copy())

        results = model(frame, verbose=False)
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    detecciones.append(
                        {
                            "frame": frame_count,
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "confidence": round(conf, 4),
                        }
                    )

    cap.release()

    df_detecciones = pd.DataFrame(detecciones)
    if not df_detecciones.empty:
        # Keep the strongest ball detection per frame to avoid duplicate-frame
        # trajectories downstream (breaks velocity/acceleration derivatives).
        df_detecciones = (
            df_detecciones.sort_values("confidence", ascending=False)
            .drop_duplicates(subset=["frame"], keep="first")
            .sort_values("frame")
            .reset_index(drop=True)
        )
    return frames_raw, df_detecciones, video_info
