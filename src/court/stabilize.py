from __future__ import annotations

import cv2
import numpy as np

from court.infer import predict_keypoints


def estimate_stable_keypoints(
    video_path: str,
    model,
    device: str,
    num_frames: int = 30,
    stride: int = 1,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for stabilization: {video_path}")

    preds = []
    frame_idx = 0
    taken = 0
    while taken < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(stride, 1) == 0:
            preds.append(predict_keypoints(model, frame, device))
            taken += 1
        frame_idx += 1

    cap.release()
    if not preds:
        raise RuntimeError("No frames available for court stabilization")

    return np.median(np.stack(preds, axis=0), axis=0).astype(np.float32)


def ema_update(prev: np.ndarray, new: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return (1.0 - alpha) * prev + alpha * new
