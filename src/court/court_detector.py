from __future__ import annotations

import cv2
import numpy as np
import torch

from .postprocess import postprocess
from .tracknet import BallTrackerNet


class TennisCourtDetector:
    def __init__(self, model_path: str, device: str = "cpu", input_size: tuple[int, int] = (640, 360)) -> None:
        self.device = device
        self.input_w, self.input_h = input_size

        self.model = BallTrackerNet(out_channels=15)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[float | None, float | None]]:
        h_orig, w_orig = frame_bgr.shape[:2]

        img = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(tensor)[0]
            pred = torch.sigmoid(pred)

        pred_np = pred.detach().cpu().numpy()

        keypoints: list[tuple[float | None, float | None]] = []
        for kps_idx in range(14):
            heatmap = (pred_np[kps_idx] * 255.0).astype(np.uint8)
            x_r, y_r = postprocess(heatmap)

            if x_r is None or y_r is None:
                keypoints.append((None, None))
                continue

            x = float(x_r) * (float(w_orig) / float(self.input_w))
            y = float(y_r) * (float(h_orig) / float(self.input_h))
            keypoints.append((x, y))

        return keypoints

    def predict(self, frame_bgr: np.ndarray) -> list[tuple[float | None, float | None]]:
        """Backward-compatible alias."""
        return self.detect(frame_bgr)

    def draw(self, frame_bgr: np.ndarray, points: list[tuple[float | None, float | None]]) -> np.ndarray:
        out = frame_bgr.copy()
        for x, y in points:
            if x is None or y is None:
                continue
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 255), -1)
        return out
