from __future__ import annotations

import cv2
import numpy as np


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def _point_in_poly(point, poly):
    return cv2.pointPolygonTest(poly.astype(np.float32), point, False) >= 0


def assign_near_far_players(player_bboxes, court_keypoints):
    if court_keypoints is None or len(court_keypoints) < 4:
        return None, None

    k = np.asarray(court_keypoints, dtype=np.float32)
    far_poly = np.array([k[0], k[1], k[9], k[8]], dtype=np.float32)
    near_poly = np.array([k[10], k[11], k[3], k[2]], dtype=np.float32)
    net_y = float((k[8][1] + k[9][1] + k[10][1] + k[11][1]) / 4.0)

    far_candidates = []
    near_candidates = []
    for bbox in player_bboxes:
        c = _center(bbox)
        if _point_in_poly(c, far_poly) or (k[0][0] <= c[0] <= k[1][0] and c[1] <= net_y):
            far_candidates.append((c[1], bbox))
        elif _point_in_poly(c, near_poly) or (k[2][0] <= c[0] <= k[3][0] and c[1] >= net_y):
            near_candidates.append((c[1], bbox))

    far_player = min(far_candidates, key=lambda x: x[0])[1] if far_candidates else None
    near_player = max(near_candidates, key=lambda x: x[0])[1] if near_candidates else None
    return near_player, far_player
