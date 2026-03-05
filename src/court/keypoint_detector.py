from __future__ import annotations

import numpy as np

from court.infer import predict_keypoints
from court.model import load_keypoints_model


class CourtKeypointDetector:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        self.model = load_keypoints_model(model_path, device)

    def predict(self, frame: np.ndarray, scale_to_frame: bool = True) -> np.ndarray:
        kpts = predict_keypoints(self.model, frame, self.device)
        if not scale_to_frame:
            return kpts
        return kpts.astype(np.int32)
