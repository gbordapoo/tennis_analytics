from __future__ import annotations

import cv2
import numpy as np

Point = tuple[float | None, float | None]

# 14-point template matching TennisCourtDetector keypoint ordering.
CANONICAL_COURT_POINTS: list[tuple[float, float]] = [
    (0.10, 0.12),
    (0.90, 0.12),
    (0.10, 0.88),
    (0.90, 0.88),
    (0.22, 0.12),
    (0.78, 0.12),
    (0.22, 0.88),
    (0.78, 0.88),
    (0.22, 0.50),
    (0.78, 0.50),
    (0.50, 0.12),
    (0.50, 0.88),
    (0.50, 0.50),
    (0.50, 0.30),
]


def fit_homography(
    pred_points: list[Point], ref_points: list[tuple[float, float]]
) -> np.ndarray | None:
    src = []
    dst = []
    for pred, ref in zip(pred_points, ref_points):
        x, y = pred
        if x is None or y is None:
            continue
        src.append(ref)
        dst.append((x, y))

    if len(src) < 4:
        return None

    src_np = np.array(src, dtype=np.float32)
    dst_np = np.array(dst, dtype=np.float32)
    H, _ = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 5.0)
    return H


def reproject_reference(H: np.ndarray | None, ref_points: list[tuple[float, float]]) -> list[Point]:
    if H is None:
        return [(None, None) for _ in ref_points]

    pts = np.array(ref_points, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in proj]


def correct_keypoints_with_homography(
    pred_points: list[Point], reproj_points: list[Point], max_shift_px: float = 35
) -> list[Point]:
    corrected: list[Point] = []
    for pred, reproj in zip(pred_points, reproj_points):
        px, py = pred
        rx, ry = reproj
        if rx is None or ry is None:
            corrected.append(pred)
            continue

        if px is None or py is None:
            corrected.append((rx, ry))
            continue

        dist = float(np.hypot(px - rx, py - ry))
        if dist > max_shift_px:
            corrected.append((rx, ry))
        else:
            corrected.append(pred)

    return corrected
