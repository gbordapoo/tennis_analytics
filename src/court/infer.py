from __future__ import annotations

import numpy as np
import torch

from court.keypoints import predict_court_keypoints


def preprocess(frame: np.ndarray) -> torch.Tensor:
    raise NotImplementedError("Use court.keypoints.predict_court_keypoints directly.")


def infer_output_domain(coords: np.ndarray) -> str:
    xmin, xmax = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
    ymin, ymax = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
    if xmin >= 0 and xmax <= 1.5 and ymin >= 0 and ymax <= 1.5:
        return "[0..1]"
    if xmin >= -2.5 and xmax <= 2.5 and ymin >= -2.5 and ymax <= 2.5:
        return "[-1..1]"
    if xmax <= 224 * 1.25 and ymax <= 224 * 1.25:
        return "input_px"
    return "image_px"


def predict_keypoints(model: torch.nn.Module, frame: np.ndarray, device: str | None = None) -> np.ndarray:
    _ = device
    return predict_court_keypoints(model, frame)
