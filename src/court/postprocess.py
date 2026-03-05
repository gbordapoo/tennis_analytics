from __future__ import annotations

import cv2
import numpy as np


def postprocess(heatmap: np.ndarray, low_thresh: int = 170, max_radius: int = 25) -> tuple[int | None, int | None]:
    """Return keypoint center from a uint8 heatmap in heatmap coordinates."""
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)

    _, binary = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    if not np.any(binary):
        return (None, None)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return (None, None)

    best_score = -1
    best_xy: tuple[int | None, int | None] = (None, None)

    for lbl in range(1, num_labels):
        x, y, w, h, _area = stats[lbl]
        radius = max(w, h) / 2.0
        if radius > max_radius:
            continue

        mask = labels == lbl
        if not np.any(mask):
            continue

        local_vals = heatmap[mask]
        score = int(local_vals.max())
        if score <= best_score:
            continue

        ys, xs = np.where(mask)
        local_idx = int(np.argmax(heatmap[ys, xs]))
        best_xy = (int(xs[local_idx]), int(ys[local_idx]))
        best_score = score

    return best_xy
