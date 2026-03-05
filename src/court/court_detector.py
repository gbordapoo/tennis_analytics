from __future__ import annotations

import cv2
import numpy as np
import torch

from .tracknet import BallTrackerNet


class TennisCourtDetector:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device

        self.model = BallTrackerNet(out_channels=15)
        sd = torch.load(model_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=True)
        self.model.to(device)
        self.model.eval()

    def predict(self, frame_bgr: np.ndarray) -> list[tuple[float, float]]:
        h_orig, w_orig = frame_bgr.shape[:2]

        img = cv2.resize(frame_bgr, (640, 360))
        img = img.astype(np.float32) / 255.0
        img = np.rollaxis(img, 2, 0)
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            out = torch.sigmoid(out)

        heatmaps = out[0].cpu().numpy()

        keypoints: list[tuple[float, float]] = []
        for i in range(14):
            idx = heatmaps[i].argmax()
            y, x = np.unravel_index(idx, heatmaps[i].shape)
            x0 = float(x) * (float(w_orig) / 640.0)
            y0 = float(y) * (float(h_orig) / 360.0)
            keypoints.append((x0, y0))

        return keypoints

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[float, float]]:
        return self.predict(frame_bgr)

    def draw(self, frame_bgr: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
        out = frame_bgr.copy()
        for x, y in points:
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 255), -1)
        return out
