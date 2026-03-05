from __future__ import annotations

import cv2
import numpy as np


def render_frame(
    frame: np.ndarray,
    court_keypoints: np.ndarray | None,
    near_player: tuple[float, float, float, float] | None,
    far_player: tuple[float, float, float, float] | None,
    ball_center: tuple[float, float] | None,
) -> np.ndarray:
    canvas = frame.copy()

    if court_keypoints is not None:
        for x, y in np.asarray(court_keypoints, dtype=np.float32):
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 255), -1)

    if near_player is not None:
        x1, y1, x2, y2 = map(int, near_player)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(canvas, "near", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if far_player is not None:
        x1, y1, x2, y2 = map(int, far_player)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(canvas, "far", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if ball_center is not None:
        cv2.circle(canvas, (int(ball_center[0]), int(ball_center[1])), 5, (0, 165, 255), -1)

    return canvas
