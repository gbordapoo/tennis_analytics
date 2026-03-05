from __future__ import annotations

import cv2
import numpy as np

# 14 canonical tennis points (meters) in fixed order.
# Origin at far-left doubles baseline corner; x rightward, y toward near side.
COURT_TEMPLATE_14 = np.array(
    [
        [0.00, 0.00],    # 0 far-left doubles corner
        [10.97, 0.00],   # 1 far-right doubles corner
        [0.00, 23.77],   # 2 near-left doubles corner
        [10.97, 23.77],  # 3 near-right doubles corner
        [1.37, 0.00],    # 4 far-left singles corner
        [9.60, 0.00],    # 5 far-right singles corner
        [1.37, 23.77],   # 6 near-left singles corner
        [9.60, 23.77],   # 7 near-right singles corner
        [1.37, 5.485],   # 8 far-left service intersection
        [9.60, 5.485],   # 9 far-right service intersection
        [1.37, 18.285],  # 10 near-left service intersection
        [9.60, 18.285],  # 11 near-right service intersection
        [5.485, 5.485],  # 12 far service T
        [5.485, 18.285], # 13 near service T
    ],
    dtype=np.float32,
)

HOMOGRAPHY_FIT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]


def fit_homography(template_pts: np.ndarray, image_pts: np.ndarray):
    H, mask = cv2.findHomography(template_pts, image_pts, method=cv2.RANSAC)
    return H, mask


def reproject(template_pts_all: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(template_pts_all.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def refine_keypoints(pred_kpts: np.ndarray, template_pts: np.ndarray = COURT_TEMPLATE_14) -> np.ndarray:
    if pred_kpts.shape[0] != template_pts.shape[0]:
        return pred_kpts
    H, _ = fit_homography(template_pts[HOMOGRAPHY_FIT_INDICES], pred_kpts[HOMOGRAPHY_FIT_INDICES])
    if H is None:
        return pred_kpts
    try:
        return reproject(template_pts, H).astype(np.float32)
    except cv2.error:
        return pred_kpts
