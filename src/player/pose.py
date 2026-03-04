from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10


def _bbox_center(person: dict[str, float]) -> tuple[float, float]:
    return (0.5 * (float(person["x1"]) + float(person["x2"])), 0.5 * (float(person["y1"]) + float(person["y2"])))


def _foot_point(person: dict[str, float]) -> tuple[float, float]:
    return (0.5 * (float(person["x1"]) + float(person["x2"])), float(person["y2"]))


def _foot_x(person: dict[str, float]) -> float:
    return _foot_point(person)[0]


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
    bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0



def select_near_far_people(
    candidates: list[dict[str, float]],
    W: int,
    H: int,
    prev_near: dict[str, float] | None = None,
    prev_far: dict[str, float] | None = None,
    cfg: dict[str, float | int] | None = None,
) -> tuple[dict[str, float] | None, dict[str, float] | None, dict[str, float | bool]]:
    """Select near/far players with strict x-gating and temporal stability."""
    if not candidates:
        return None, None, {"gate_fallback": False, "gate_left_px": 0.0, "gate_right_px": float(W)}

    cfg = cfg or {}
    top_k = int(cfg.get("top_k", 10))
    xgate_left_frac = float(cfg.get("player_xgate_left", 0.18))
    xgate_right_frac = float(cfg.get("player_xgate_right", 0.82))
    temporal_max_dist_frac = float(cfg.get("temporal_max_dist_frac", 0.25))
    min_conf = float(cfg.get("player_min_conf", 0.3))

    conf_filtered = [c for c in candidates if float(c.get("conf", 0.0)) >= min_conf]
    if not conf_filtered:
        return None, None, {"gate_fallback": True, "gate_left_px": 0.0, "gate_right_px": float(W)}

    sorted_by_conf = sorted(conf_filtered, key=lambda item: float(item["conf"]), reverse=True)[: max(2, top_k)]

    cal_far_left_x = cfg.get("calibration_far_left_x")
    cal_far_right_x = cfg.get("calibration_far_right_x")
    cal_margin_px = float(cfg.get("calibration_x_margin_px", 20.0))

    if cal_far_left_x is not None and cal_far_right_x is not None:
        min_foot_x = min(float(cal_far_left_x), float(cal_far_right_x)) - cal_margin_px
        max_foot_x = max(float(cal_far_left_x), float(cal_far_right_x)) + cal_margin_px
    else:
        min_foot_x = min(xgate_left_frac, xgate_right_frac) * float(W)
        max_foot_x = max(xgate_left_frac, xgate_right_frac) * float(W)

    gated = [p for p in sorted_by_conf if min_foot_x <= _foot_x(p) <= max_foot_x]
    gate_fallback = len(gated) == 0
    pool = gated if gated else sorted_by_conf

    if len(pool) == 1:
        return pool[0], None, {"gate_fallback": gate_fallback, "gate_left_px": min_foot_x, "gate_right_px": max_foot_x}

    near_default = max(pool, key=lambda item: _bbox_center(item)[1])
    far_default = min((p for p in pool if p is not near_default), key=lambda item: _bbox_center(item)[1], default=None)

    max_temporal_dist = temporal_max_dist_frac * float(np.hypot(W, H))

    def _prefer_temporal(
        prev_person: dict[str, float] | None,
        fallback: dict[str, float] | None,
        options: list[dict[str, float]],
    ) -> dict[str, float] | None:
        if fallback is None:
            return None
        if prev_person is None:
            return fallback

        prev_center = _bbox_center(prev_person)
        ranked = sorted(
            options,
            key=lambda p: (
                -_bbox_iou(p, prev_person),
                _distance(_bbox_center(p), prev_center),
            ),
        )
        best = ranked[0] if ranked else fallback
        if _distance(_bbox_center(best), prev_center) <= max_temporal_dist:
            return best
        return fallback

    near = _prefer_temporal(prev_near, near_default, pool)
    far_options = [p for p in pool if p is not near]
    far = _prefer_temporal(prev_far, far_default, far_options) if far_options else None

    if near is not None and far is near:
        alternatives = [p for p in pool if p is not near]
        far = min(alternatives, key=lambda item: _bbox_center(item)[1], default=None)

    debug = {
        "gate_fallback": gate_fallback,
        "gate_left_px": float(min_foot_x),
        "gate_right_px": float(max_foot_x),
    }
    return near, far, debug


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
        "gate_fallback",
        "gate_left_px",
        "gate_right_px",
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
        near_person, far_person, debug_info = select_near_far_people(
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
                    "gate_fallback": bool(debug_info.get("gate_fallback", False)),
                    "gate_left_px": float(debug_info.get("gate_left_px", np.nan)),
                    "gate_right_px": float(debug_info.get("gate_right_px", np.nan)),
                }
            )
        prev_near = near_person if near_person is not None else prev_near
        prev_far = far_person if far_person is not None else prev_far

    return pd.DataFrame(rows, columns=columns)
