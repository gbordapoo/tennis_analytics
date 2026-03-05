from __future__ import annotations

import cv2
import numpy as np
import torch

from .homography_refine import CANONICAL_COURT_POINTS, reproject_reference
from .postprocess import apply_homography_gated, postprocess_heatmap, refine_kps
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
        low_conf_threshold: float = 0.35,
    ) -> None:
        self.device = device
        self.refine_lines = refine_lines
        self.refine_homography = refine_homography
        self.crop_size = crop_size
        self.max_shift_px = max_shift_px
        self.low_conf_threshold = low_conf_threshold

        self.last_raw_keypoints: list[Point] = []
        self.last_refined_keypoints: list[Point] = []
        self.last_final_keypoints: list[Point] = []
        self.last_homography_used: bool = False
        self.last_homography_replacements: int = 0

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
        confidences: list[float] = []
        sx = float(w_orig) / 640.0
        sy = float(h_orig) / 360.0
        hm_h, hm_w = heatmaps[0].shape
        hm_sx = 640.0 / float(hm_w)
        hm_sy = 360.0 / float(hm_h)

        for i in range(14):
            hm = heatmaps[i]
            hm_u8 = (hm * 255).astype(np.uint8)
            x_pp, y_pp = postprocess_heatmap(hm_u8, scale=2)
            if x_pp is None or y_pp is None:
                idx = hm.argmax()
                y_hm, x_hm = np.unravel_index(idx, hm.shape)
                x_640 = float(x_hm) * hm_sx
                y_360 = float(y_hm) * hm_sy
            else:
                x_640 = float(x_pp)
                y_360 = float(y_pp)

            raw_keypoints.append((x_640 * sx, y_360 * sy))
            confidences.append(float(hm.max()))

        refined_keypoints = list(raw_keypoints)
        if self.refine_lines:
            refined_keypoints = []
            for x, y in raw_keypoints:
                if x is None or y is None:
                    refined_keypoints.append((x, y))
                    continue
                rx, ry = refine_kps(frame_bgr, x, y, crop_size=self.crop_size)
                if rx is None or ry is None:
                    refined_keypoints.append((x, y))
                else:
                    refined_keypoints.append((rx, ry))

        final_keypoints = list(refined_keypoints)
        homography_used = False
        homography_replacements = 0

        if self.refine_homography:
            anchor_src: list[tuple[float, float]] = []
            anchor_dst: list[tuple[float, float]] = []
            for idx, ((px, py), ref) in enumerate(zip(refined_keypoints, CANONICAL_COURT_POINTS)):
                if px is None or py is None:
                    continue
                if confidences[idx] < self.low_conf_threshold:
                    continue
                anchor_src.append(ref)
                anchor_dst.append((float(px), float(py)))

            H = None
            if len(anchor_src) >= 4:
                H, _ = cv2.findHomography(
                    np.array(anchor_src, dtype=np.float32),
                    np.array(anchor_dst, dtype=np.float32),
                    cv2.RANSAC,
                    5.0,
                )

            if H is not None:
                homography_used = True
                reproj = reproject_reference(H, CANONICAL_COURT_POINTS)
                final_keypoints, homography_replacements = apply_homography_gated(
                    points=refined_keypoints,
                    confidences=confidences,
                    reproj_points=reproj,
                    frame_shape=(h_orig, w_orig),
                    max_shift_px=self.max_shift_px,
                    low_conf_threshold=self.low_conf_threshold,
                )

        final_keypoints = final_keypoints[:14]
        if len(final_keypoints) < 14:
            final_keypoints.extend(raw_keypoints[len(final_keypoints) : 14])

        self.last_raw_keypoints = raw_keypoints[:14]
        self.last_refined_keypoints = refined_keypoints[:14]
        self.last_final_keypoints = final_keypoints[:14]
        self.last_homography_used = homography_used
        self.last_homography_replacements = homography_replacements

        return final_keypoints[:14]

    def detect(self, frame_bgr: np.ndarray) -> list[Point]:
        return self.predict(frame_bgr)

    def draw(self, frame_bgr: np.ndarray, points: list[Point]) -> np.ndarray:
        out = frame_bgr.copy()
        for x, y in points:
            if x is None or y is None:
                continue
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 255), -1)
        return out
