from __future__ import annotations

import cv2
import numpy as np
import torch

from .homography_refine import (
    CANONICAL_COURT_POINTS,
    correct_keypoints_with_homography,
    fit_homography,
    reproject_reference,
)
from .postprocess import postprocess_heatmap, refine_kps
from .tracknet import BallTrackerNet


Point = tuple[float | None, float | None]


class TennisCourtDetector:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        refine_lines: bool = True,
        refine_homography: bool = True,
        crop_size: int = 40,
        max_shift_px: float = 35,
    ) -> None:
        self.device = device
        self.refine_lines = refine_lines
        self.refine_homography = refine_homography
        self.crop_size = crop_size
        self.max_shift_px = max_shift_px

        self.last_raw_keypoints: list[Point] = []
        self.last_refined_keypoints: list[Point] = []
        self.last_final_keypoints: list[Point] = []

        self.model = BallTrackerNet(out_channels=15)
        sd = torch.load(model_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=True)
        self.model.to(device)
        self.model.eval()

    def predict(self, frame_bgr: np.ndarray) -> list[Point]:
        h_orig, w_orig = frame_bgr.shape[:2]

        img = cv2.resize(frame_bgr, (640, 360))
        img = img.astype(np.float32) / 255.0
        img = np.rollaxis(img, 2, 0)
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            out = torch.sigmoid(out)

        heatmaps = out[0].cpu().numpy()

        raw_keypoints: list[Point] = []
        sx = float(w_orig) / 640.0
        sy = float(h_orig) / 360.0
        for i in range(14):
            hm_u8 = (heatmaps[i] * 255).astype(np.uint8)
            x_pp, y_pp = postprocess_heatmap(hm_u8, scale=2)
            if x_pp is None or y_pp is None:
                idx = heatmaps[i].argmax()
                y, x = np.unravel_index(idx, heatmaps[i].shape)
                x0 = float(x) * sx
                y0 = float(y) * sy
            else:
                x0 = float(x_pp) * sx
                y0 = float(y_pp) * sy
            raw_keypoints.append((x0, y0))

        refined_keypoints = list(raw_keypoints)
        if self.refine_lines:
            refined_keypoints = [
                refine_kps(frame_bgr, x, y, crop_size=self.crop_size) if x is not None and y is not None else (x, y)
                for x, y in raw_keypoints
            ]

        final_keypoints = list(refined_keypoints)
        if self.refine_homography:
            H = fit_homography(refined_keypoints, CANONICAL_COURT_POINTS)
            reproj = reproject_reference(H, CANONICAL_COURT_POINTS)
            final_keypoints = correct_keypoints_with_homography(
                refined_keypoints,
                reproj,
                max_shift_px=self.max_shift_px,
            )

        self.last_raw_keypoints = raw_keypoints
        self.last_refined_keypoints = refined_keypoints
        self.last_final_keypoints = final_keypoints

        return final_keypoints

    def detect(self, frame_bgr: np.ndarray) -> list[Point]:
        return self.predict(frame_bgr)

    def draw(self, frame_bgr: np.ndarray, points: list[Point]) -> np.ndarray:
        out = frame_bgr.copy()
        for x, y in points:
            if x is None or y is None:
                continue
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 255), -1)
        return out
