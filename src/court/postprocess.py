from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np


Line = tuple[int, int, int, int]
Point = tuple[float, float]


def postprocess_heatmap(
    heatmap: np.ndarray,
    scale: float = 2,
    low_thresh: int = 155,
    min_radius: int = 10,
    max_radius: int = 30,
) -> tuple[float | None, float | None]:
    _, binary = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None or circles.size == 0:
        return None, None

    x, y, _ = circles[0][0]
    return float(x * scale), float(y * scale)


def detect_lines(image: np.ndarray) -> list[Line]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=3)
    if lines is None:
        return []
    return [tuple(int(v) for v in line[0]) for line in lines]


def _line_distance(a: Line, b: Line) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    d1 = np.hypot(ax1 - bx1, ay1 - by1) + np.hypot(ax2 - bx2, ay2 - by2)
    d2 = np.hypot(ax1 - bx2, ay1 - by2) + np.hypot(ax2 - bx1, ay2 - by1)
    return float(min(d1, d2))


def merge_lines(lines: list[Line]) -> list[Line]:
    if len(lines) <= 1:
        return lines

    merged: list[Line] = []
    used = [False] * len(lines)
    for i, line in enumerate(lines):
        if used[i]:
            continue
        group = [line]
        used[i] = True
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            if _line_distance(line, lines[j]) < 20:
                group.append(lines[j])
                used[j] = True

        x1 = int(np.mean([g[0] for g in group]))
        y1 = int(np.mean([g[1] for g in group]))
        x2 = int(np.mean([g[2] for g in group]))
        y2 = int(np.mean([g[3] for g in group]))
        merged.append((x1, y1, x2, y2))

    return merged


def line_intersection(line1: Line, line2: Line) -> Point | None:
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
    py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
    return float(px), float(py)


def refine_kps(img: np.ndarray, x_ct: float, y_ct: float, crop_size: int = 40) -> tuple[float, float]:
    h, w = img.shape[:2]
    half = crop_size // 2

    x0 = max(0, int(round(x_ct)) - half)
    y0 = max(0, int(round(y_ct)) - half)
    x1 = min(w, x0 + crop_size)
    y1 = min(h, y0 + crop_size)

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return x_ct, y_ct

    lines = merge_lines(detect_lines(crop))
    if len(lines) < 2:
        return x_ct, y_ct

    best: Point | None = None
    center = (crop.shape[1] / 2.0, crop.shape[0] / 2.0)
    best_dist = float("inf")
    for line1, line2 in combinations(lines, 2):
        pt = line_intersection(line1, line2)
        if pt is None:
            continue
        x, y = pt
        if not (0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]):
            continue
        dist = float(np.hypot(x - center[0], y - center[1]))
        if dist < best_dist:
            best = pt
            best_dist = dist

    if best is None:
        return x_ct, y_ct

    return float(x0 + best[0]), float(y0 + best[1])
