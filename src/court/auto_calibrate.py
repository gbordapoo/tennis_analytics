from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from court.homography import compute_homography

SINGLES_WIDTH_M = 8.23
SINGLES_LENGTH_M = 23.77


def _line_score(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))
    white_ratio = float(np.count_nonzero(white_mask)) / float(white_mask.size)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)

    return (white_ratio * 0.7) + (edge_ratio * 0.3)


def pick_best_frame(video_path: Path, sample_every: int = 10, max_frames: int = 300) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    best_frame: np.ndarray | None = None
    best_score = -1.0
    frame_idx = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every == 0:
            score = _line_score(frame)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()

        frame_idx += 1

    cap.release()

    if best_frame is None:
        raise RuntimeError("Could not sample any frame from video")
    return best_frame


def _order_points_near_far(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    y_sorted = points[np.argsort(points[:, 1])]

    near = y_sorted[2:]
    far = y_sorted[:2]

    near_left, near_right = near[np.argsort(near[:, 0])]
    far_left, far_right = far[np.argsort(far[:, 0])]

    return np.asarray([near_left, near_right, far_left, far_right], dtype=np.float32)


def detect_court_corners(frame: np.ndarray) -> tuple[np.ndarray, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 85, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(clean, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=20)

    if lines is None or len(lines) < 4:
        return np.empty((0, 2), dtype=np.float32), 0.0

    long_points: list[list[float]] = []
    line_lengths: list[float] = []
    for item in lines[:, 0, :]:
        x1, y1, x2, y2 = item
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length >= 100:
            long_points.append([x1, y1])
            long_points.append([x2, y2])
            line_lengths.append(length)

    if len(long_points) < 8:
        return np.empty((0, 2), dtype=np.float32), 0.0

    pts = np.asarray(long_points, dtype=np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(pts)
        quad = cv2.boxPoints(rect).astype(np.float32)

    quad = _order_points_near_far(quad)

    avg_len = float(np.mean(line_lengths)) if line_lengths else 0.0
    lines_conf = min(1.0, len(line_lengths) / 20.0)
    length_conf = min(1.0, avg_len / 300.0)
    confidence = max(0.0, min(1.0, 0.5 * lines_conf + 0.5 * length_conf))

    return quad, confidence


def run_static_auto_calibration(video_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    best_frame = pick_best_frame(video_path)
    pixel_points, confidence = detect_court_corners(best_frame)

    if pixel_points.shape != (4, 2):
        return best_frame, np.empty((0, 2), dtype=np.float32), np.empty((0, 0), dtype=np.float32), 0.0

    world_points = np.asarray(
        [
            [0.0, 0.0],
            [SINGLES_WIDTH_M, 0.0],
            [0.0, SINGLES_LENGTH_M],
            [SINGLES_WIDTH_M, SINGLES_LENGTH_M],
        ],
        dtype=np.float32,
    )

    H = compute_homography(pixel_points, world_points)
    return best_frame, pixel_points, H, confidence


def draw_court_overlay(frame: np.ndarray, pixel_points: np.ndarray, output_path: Path) -> None:
    overlay = frame.copy()
    if pixel_points.shape == (4, 2):
        pts = pixel_points.astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 255), 3)
        for idx, (x, y) in enumerate(pts):
            cv2.circle(overlay, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                str(idx + 1),
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(output_path), overlay)
