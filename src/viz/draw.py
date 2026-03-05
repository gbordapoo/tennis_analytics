from __future__ import annotations

import cv2
import numpy as np

# Matches src/court/geometry.py template ordering.
COURT_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (6, 7), (4, 6), (5, 7),
    (8, 9), (10, 11), (8, 10), (9, 11),
    (12, 13),
]


def draw_keypoints(frame, keypoints: np.ndarray):
    for x, y in keypoints:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
    for i, j in COURT_EDGES:
        if i < len(keypoints) and j < len(keypoints):
            p1 = tuple(np.int32(keypoints[i]))
            p2 = tuple(np.int32(keypoints[j]))
            cv2.line(frame, p1, p2, (255, 255, 0), 2)
    return frame


def draw_players(frame, near_player=None, far_player=None):
    if near_player is not None:
        x1, y1, x2, y2 = [int(v) for v in near_player]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, "near", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    if far_player is not None:
        x1, y1, x2, y2 = [int(v) for v in far_player]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "far", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def draw_ball(frame, balls):
    for x1, y1, x2, y2 in balls:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 5, (0, 165, 255), -1)
    return frame
