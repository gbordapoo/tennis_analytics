from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10


def _bbox_center(person: dict[str, float]) -> tuple[float, float]:
    return (0.5 * (float(person["x1"]) + float(person["x2"])), 0.5 * (float(person["y1"]) + float(person["y2"])))


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def select_near_far_people(
    candidates: list[dict[str, float]],
    W: int,
    H: int,
    prev_near: dict[str, float] | None = None,
    prev_far: dict[str, float] | None = None,
    cfg: dict[str, float | int] | None = None,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Select near/far players robustly while avoiding sideline people.

    Strategy:
    - keep top-K by confidence,
    - gate by center-x in [min_cx_frac*W, max_cx_frac*W],
    - near=max(y2), far=min(y2) with optional temporal stabilization.
    """
    if not candidates:
        return None, None

    cfg = cfg or {}
    top_k = int(cfg.get("top_k", 10))
    min_cx_frac = float(cfg.get("min_cx_frac", 0.15))
    max_cx_frac = float(cfg.get("max_cx_frac", 0.85))
    temporal_max_dist_frac = float(cfg.get("temporal_max_dist_frac", 0.25))

    sorted_by_conf = sorted(candidates, key=lambda item: float(item["conf"]), reverse=True)[: max(2, top_k)]

    min_cx = min_cx_frac * float(W)
    max_cx = max_cx_frac * float(W)
    gated = [p for p in sorted_by_conf if min_cx <= _bbox_center(p)[0] <= max_cx]
    pool = gated if len(gated) >= 2 else sorted_by_conf

    if len(pool) == 1:
        return pool[0], None

    near_default = max(pool, key=lambda item: float(item["y2"]))
    far_default = min((p for p in pool if p is not near_default), key=lambda item: float(item["y2"]), default=None)

    if prev_near is None and prev_far is None:
        return near_default, far_default

    max_temporal_dist = temporal_max_dist_frac * float(np.hypot(W, H))

    def _closest(prev_person: dict[str, float] | None, fallback: dict[str, float] | None, banned: set[int]) -> dict[str, float] | None:
        if prev_person is None:
            return fallback
        prev_center = _bbox_center(prev_person)
        options = [p for p in pool if id(p) not in banned]
        if not options:
            return fallback
        best = min(options, key=lambda p: _distance(_bbox_center(p), prev_center))
        if _distance(_bbox_center(best), prev_center) <= max_temporal_dist:
            return best
        return fallback

    near = _closest(prev_near, near_default, banned=set())
    banned_ids = {id(near)} if near is not None else set()
    far = _closest(prev_far, far_default, banned=banned_ids)

    if near is not None and far is near:
        alternatives = [p for p in pool if p is not near]
        far = min(alternatives, key=lambda item: float(item["y2"]), default=None)

    return near, far


def ensure_pose_model(weights_path: Path) -> Path:
    """Ensure pose weights exist locally and never auto-download them."""
    target = Path(weights_path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(
            "Pose model not found at models/yolov8n-pose.pt. "
            "Download it and place it in models/."
        )

    return target


def _extract_wrist(keypoints_xy: Any, keypoints_conf: Any, index: int) -> tuple[float, float]:
    if keypoints_xy is None:
        return (np.nan, np.nan)

    try:
        point = keypoints_xy[index]
    except (IndexError, TypeError):
        return (np.nan, np.nan)

    conf_val = None
    if keypoints_conf is not None:
        try:
            conf_val = float(keypoints_conf[index])
        except (IndexError, TypeError, ValueError):
            conf_val = None

    if conf_val is not None and conf_val <= 0:
        return (np.nan, np.nan)

    try:
        return (float(point[0]), float(point[1]))
    except (TypeError, ValueError, IndexError):
        return (np.nan, np.nan)


def detect_players(
    frames_raw: list,
    weights_path: str | Path = "models/yolov8n-pose.pt",
    selection_cfg: dict[str, float | int] | None = None,
) -> pd.DataFrame:
    """Detect up to two players per frame and assign near/far labels."""
    columns = [
        "frame",
        "player",
        "x1",
        "y1",
        "x2",
        "y2",
        "wrist_l_x",
        "wrist_l_y",
        "wrist_r_x",
        "wrist_r_y",
        "conf",
    ]
    if not frames_raw:
        return pd.DataFrame(columns=columns)

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    rows: list[dict[str, Any]] = []
    prev_near: dict[str, float] | None = None
    prev_far: dict[str, float] | None = None

    for frame_idx, frame in enumerate(frames_raw, start=1):
        results = model(frame, verbose=False)
        people: list[dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            keypoints = result.keypoints
            keypoints_xy = keypoints.xy.cpu().numpy() if keypoints is not None and keypoints.xy is not None else None
            keypoints_conf = keypoints.conf.cpu().numpy() if keypoints is not None and keypoints.conf is not None else None

            for person_idx, box in enumerate(boxes):
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                person_kp_xy = keypoints_xy[person_idx] if keypoints_xy is not None and person_idx < len(keypoints_xy) else None
                person_kp_conf = (
                    keypoints_conf[person_idx] if keypoints_conf is not None and person_idx < len(keypoints_conf) else None
                )

                wrist_l_x, wrist_l_y = _extract_wrist(person_kp_xy, person_kp_conf, LEFT_WRIST_INDEX)
                wrist_r_x, wrist_r_y = _extract_wrist(person_kp_xy, person_kp_conf, RIGHT_WRIST_INDEX)

                people.append(
                    {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "wrist_l_x": wrist_l_x,
                        "wrist_l_y": wrist_l_y,
                        "wrist_r_x": wrist_r_x,
                        "wrist_r_y": wrist_r_y,
                        "conf": conf,
                    }
                )

        if not people:
            continue

        H, W = frame.shape[:2]
        near_person, far_person = select_near_far_people(
            candidates=people,
            W=int(W),
            H=int(H),
            prev_near=prev_near,
            prev_far=prev_far,
            cfg=selection_cfg,
        )

        selected = [("near", near_person), ("far", far_person)]
        for label, person in selected:
            if person is None:
                continue
            rows.append(
                {
                    "frame": frame_idx,
                    "player": label,
                    **person,
                }
            )
        prev_near = near_person if near_person is not None else prev_near
        prev_far = far_person if far_person is not None else prev_far

    return pd.DataFrame(rows, columns=columns)
