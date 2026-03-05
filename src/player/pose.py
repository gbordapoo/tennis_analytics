from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10


def _bbox_center(person: dict[str, float]) -> tuple[float, float]:
    return (0.5 * (float(person["x1"]) + float(person["x2"])), 0.5 * (float(person["y1"]) + float(person["y2"])))


def _foot_point(person: dict[str, float]) -> tuple[float, float]:
    return (0.5 * (float(person["x1"]) + float(person["x2"])), float(person["y2"]))


def foot_point(bbox: dict[str, float]) -> tuple[float, float]:
    return _foot_point(bbox)


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


def choose_players_by_keypoints(
    court_keypoints: list[float] | None,
    player_dict_frame: dict[int, dict[str, float]],
    band_margin_px: float = 80.0,
) -> list[int]:
    if not player_dict_frame:
        return []

    if court_keypoints is None or len(court_keypoints) < 4:
        return list(player_dict_frame.keys())[:2]

    kp_pairs = [(float(court_keypoints[i]), float(court_keypoints[i + 1])) for i in range(0, min(len(court_keypoints), 28), 2)]
    ys = [p[1] for p in kp_pairs]
    band_min = min(ys) - float(band_margin_px)
    band_max = max(ys) + float(band_margin_px)

    ranking: list[tuple[float, int]] = []
    for track_id, bbox in player_dict_frame.items():
        cx = 0.5 * (float(bbox["x1"]) + float(bbox["x2"]))
        cy = 0.5 * (float(bbox["y1"]) + float(bbox["y2"]))
        foot_y = float(bbox["y2"])
        if foot_y < band_min or foot_y > band_max:
            continue
        min_dist = min(_distance((cx, cy), kp) for kp in kp_pairs)
        ranking.append((min_dist, int(track_id)))

    if len(ranking) < 2:
        for track_id, bbox in player_dict_frame.items():
            cx = 0.5 * (float(bbox["x1"]) + float(bbox["x2"]))
            cy = 0.5 * (float(bbox["y1"]) + float(bbox["y2"]))
            min_dist = min(_distance((cx, cy), kp) for kp in kp_pairs)
            ranking.append((min_dist, int(track_id)))

    ranking.sort(key=lambda item: item[0])
    chosen: list[int] = []
    for _, tid in ranking:
        if tid not in chosen:
            chosen.append(tid)
        if len(chosen) == 2:
            break
    return chosen


def filter_to_chosen_ids(
    player_dict_frame: dict[int, dict[str, float]],
    chosen_ids: list[int],
) -> dict[int, dict[str, float]]:
    return {tid: person for tid, person in player_dict_frame.items() if tid in chosen_ids}


def label_far_near(player_dict: dict[int, dict[str, float]]) -> tuple[int | None, int | None]:
    if not player_dict:
        return None, None
    ordered = sorted(player_dict.items(), key=lambda item: float(item[1]["y2"]))
    if len(ordered) == 1:
        return int(ordered[0][0]), None
    far_id = int(ordered[0][0])
    near_id = int(ordered[-1][0])
    if far_id == near_id:
        return far_id, None
    return far_id, near_id


def point_in_poly(point: tuple[float, float], poly: np.ndarray | None) -> bool:
    if poly is None:
        return False
    poly_np = np.asarray(poly, dtype=np.float32)
    if poly_np.ndim != 2 or poly_np.shape[1] != 2 or len(poly_np) < 3:
        return False
    return cv2.pointPolygonTest(poly_np, point, False) >= 0


def _expand_polygon(poly: np.ndarray, margin_px: float) -> np.ndarray:
    if margin_px <= 0:
        return poly.astype(np.float32)
    center = np.mean(poly, axis=0)
    vecs = poly - center
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = np.divide(vecs, np.maximum(norms, 1e-6))
    return (poly + unit * float(margin_px)).astype(np.float32)


def build_halfcourt_polys(keypoints: np.ndarray | list[list[float]] | None, margin_px: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    if keypoints is None:
        return None, None
    pts = np.asarray(keypoints, dtype=np.float32)
    if pts.shape != (4, 2):
        return None, None

    near_left, near_right, far_left, far_right = pts
    mid_left = 0.5 * (near_left + far_left)
    mid_right = 0.5 * (near_right + far_right)

    far_poly = np.asarray([far_left, far_right, mid_right, mid_left], dtype=np.float32)
    near_poly = np.asarray([mid_left, mid_right, near_right, near_left], dtype=np.float32)
    return _expand_polygon(far_poly, margin_px), _expand_polygon(near_poly, margin_px)


def stabilize_selection(
    prev_id: int | None,
    current_candidates_dict: dict[int, dict[str, float]],
    poly: np.ndarray | None,
) -> int | None:
    if prev_id is None or prev_id not in current_candidates_dict:
        return None
    if poly is None:
        return prev_id
    return prev_id if point_in_poly(foot_point(current_candidates_dict[prev_id]), poly) else None


def select_player(
    candidates: dict[int, dict[str, float]],
    target: str,
    baseline_center: tuple[float, float],
) -> int | None:
    if not candidates:
        return None
    heights = np.asarray([max(1.0, float(c["y2"]) - float(c["y1"])) for c in candidates.values()], dtype=np.float32)
    expected_height = float(np.median(heights)) if len(heights) else 1.0
    if expected_height <= 0:
        expected_height = float(np.max(heights)) if len(heights) else 1.0

    def _score(item: tuple[int, dict[str, float]]) -> float:
        _track_id, person = item
        px, py = foot_point(person)
        h = max(1.0, float(person["y2"]) - float(person["y1"]))
        dist = _distance((px, py), baseline_center)
        return dist + 0.001 * abs(expected_height - h)

    return min(candidates.items(), key=_score)[0]


def _legacy_select_near_far(
    candidates: list[dict[str, float]],
    W: int,
    H: int,
    prev_near: dict[str, float] | None,
    prev_far: dict[str, float] | None,
    cfg: dict[str, float | int],
) -> tuple[dict[str, float] | None, dict[str, float] | None, dict[str, float | bool]]:
    top_k = int(cfg.get("top_k", 10))
    xgate_left_frac = float(cfg.get("player_xgate_left", 0.18))
    xgate_right_frac = float(cfg.get("player_xgate_right", 0.82))
    temporal_max_dist_frac = float(cfg.get("temporal_max_dist_frac", 0.25))

    sorted_by_conf = sorted(candidates, key=lambda item: float(item["conf"]), reverse=True)[: max(2, top_k)]

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



def select_near_far_people(
    candidates: list[dict[str, float]],
    W: int,
    H: int,
    prev_near: dict[str, float] | None = None,
    prev_far: dict[str, float] | None = None,
    cfg: dict[str, float | int] | None = None,
) -> tuple[dict[str, float] | None, dict[str, float] | None, dict[str, float | bool]]:
    """Select near/far players with geometric half-court filtering and temporal stability."""
    if not candidates:
        return None, None, {"gate_fallback": False, "gate_left_px": 0.0, "gate_right_px": float(W)}

    cfg = cfg or {}
    min_conf = float(cfg.get("player_min_conf", 0.3))

    conf_filtered = [c for c in candidates if float(c.get("conf", 0.0)) >= min_conf]
    if not conf_filtered:
        return None, None, {"gate_fallback": True, "gate_left_px": 0.0, "gate_right_px": float(W)}

    keypoints = cfg.get("calibration_pixel_points")
    margin_px = float(cfg.get("player_halfcourt_margin_px", 16.0))
    far_poly, near_poly = build_halfcourt_polys(keypoints, margin_px)

    candidates_by_id = {int(person.get("track_id", idx)): person for idx, person in enumerate(conf_filtered)}

    if far_poly is None or near_poly is None:
        near, far, debug = _legacy_select_near_far(conf_filtered, W, H, prev_near, prev_far, cfg)
        debug["selection_fallback"] = True
        return near, far, debug

    far_candidates = {tid: p for tid, p in candidates_by_id.items() if point_in_poly(foot_point(p), far_poly)}
    near_candidates = {tid: p for tid, p in candidates_by_id.items() if point_in_poly(foot_point(p), near_poly)}

    prev_far_id = int(prev_far["track_id"]) if prev_far and "track_id" in prev_far else None
    prev_near_id = int(prev_near["track_id"]) if prev_near and "track_id" in prev_near else None

    far_id = stabilize_selection(prev_far_id, far_candidates, far_poly)
    near_id = stabilize_selection(prev_near_id, near_candidates, near_poly)
    if far_id is None:
        baseline_far_center = (0.5 * (float(keypoints[2][0]) + float(keypoints[3][0])), 0.5 * (float(keypoints[2][1]) + float(keypoints[3][1])))
        far_id = select_player(far_candidates, target="far", baseline_center=baseline_far_center)
    if near_id is None:
        baseline_near_center = (0.5 * (float(keypoints[0][0]) + float(keypoints[1][0])), 0.5 * (float(keypoints[0][1]) + float(keypoints[1][1])))
        near_id = select_player(near_candidates, target="near", baseline_center=baseline_near_center)

    if near_id is not None and near_id == far_id:
        near_id = None

    near = candidates_by_id.get(near_id) if near_id is not None else None
    far = candidates_by_id.get(far_id) if far_id is not None else None

    if far is None and prev_far is not None and point_in_poly(foot_point(prev_far), far_poly):
        far = prev_far
    if near is None and prev_near is not None and point_in_poly(foot_point(prev_near), near_poly):
        near = prev_near

    if far is None or near is None:
        near_fallback, far_fallback, legacy_debug = _legacy_select_near_far(conf_filtered, W, H, prev_near, prev_far, cfg)
        near = near if near is not None else near_fallback
        far = far if far is not None else far_fallback
        debug = {**legacy_debug, "selection_fallback": True}
    else:
        debug = {"gate_fallback": False, "gate_left_px": 0.0, "gate_right_px": float(W), "selection_fallback": False}

    debug.update(
        {
            "far_candidates": len(far_candidates),
            "near_candidates": len(near_candidates),
            "filtered_out": len(candidates_by_id) - len(set(far_candidates) | set(near_candidates)),
            "far_poly": far_poly,
            "near_poly": near_poly,
            "far_track_id": far.get("track_id") if far is not None else -1,
            "near_track_id": near.get("track_id") if near is not None else -1,
        }
    )
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
        "frame", "player", "x1", "y1", "x2", "y2", "wrist_l_x", "wrist_l_y", "wrist_r_x", "wrist_r_y",
        "conf", "track_id", "gate_fallback", "gate_left_px", "gate_right_px", "selection_fallback",
        "far_candidates", "near_candidates", "filtered_out", "foot_x", "foot_y", "far_poly", "near_poly",
    ]
    if not frames_raw:
        return pd.DataFrame(columns=columns)

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    rows: list[dict[str, Any]] = []
    prev_tracks: dict[int, dict[str, float]] = {}
    next_track_id = 1
    cfg = selection_cfg or {}
    min_conf = float(cfg.get("player_min_conf", 0.3))
    court_keypoints = cfg.get("court_keypoints_14")
    missing_both_frames = 0
    reselection_threshold = int(cfg.get("player_reselect_missing_frames", 15))
    chosen_ids: list[int] = []
    debug_log = bool(cfg.get("player_debug_log", False))

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
                if conf < min_conf:
                    continue

                person_kp_xy = keypoints_xy[person_idx] if keypoints_xy is not None and person_idx < len(keypoints_xy) else None
                person_kp_conf = keypoints_conf[person_idx] if keypoints_conf is not None and person_idx < len(keypoints_conf) else None
                wrist_l_x, wrist_l_y = _extract_wrist(person_kp_xy, person_kp_conf, LEFT_WRIST_INDEX)
                wrist_r_x, wrist_r_y = _extract_wrist(person_kp_xy, person_kp_conf, RIGHT_WRIST_INDEX)

                people.append({
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "wrist_l_x": wrist_l_x, "wrist_l_y": wrist_l_y, "wrist_r_x": wrist_r_x, "wrist_r_y": wrist_r_y,
                    "conf": conf,
                })

        assigned_ids: set[int] = set()
        for person in people:
            best_track = None
            best_iou = 0.0
            for track_id, prev_person in prev_tracks.items():
                if track_id in assigned_ids:
                    continue
                iou = _bbox_iou(person, prev_person)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track_id
            if best_track is not None and best_iou >= 0.3:
                person["track_id"] = best_track
                assigned_ids.add(best_track)
            else:
                person["track_id"] = next_track_id
                assigned_ids.add(next_track_id)
                next_track_id += 1
        prev_tracks = {int(p["track_id"]): p for p in people}

        people_by_id = {int(p["track_id"]): p for p in people}
        if not chosen_ids and people_by_id:
            chosen_ids = choose_players_by_keypoints(court_keypoints, people_by_id)
            if debug_log:
                print(f"[players] initial chosen_ids={chosen_ids}")

        selected_by_id = filter_to_chosen_ids(people_by_id, chosen_ids)
        if not selected_by_id:
            missing_both_frames += 1
            if missing_both_frames > reselection_threshold and people_by_id:
                chosen_ids = choose_players_by_keypoints(court_keypoints, people_by_id)
                selected_by_id = filter_to_chosen_ids(people_by_id, chosen_ids)
                missing_both_frames = 0
                if debug_log:
                    print(f"[players][frame={frame_idx}] reselection chosen_ids={chosen_ids}")
        else:
            missing_both_frames = 0

        far_id, near_id = label_far_near(selected_by_id)
        far_person = selected_by_id.get(far_id) if far_id is not None else None
        near_person = selected_by_id.get(near_id) if near_id is not None else None

        for label, person in (("far", far_person), ("near", near_person)):
            if person is None:
                continue
            fpx, fpy = foot_point(person)
            rows.append({
                "frame": frame_idx,
                "player": label,
                **person,
                "gate_fallback": False,
                "gate_left_px": np.nan,
                "gate_right_px": np.nan,
                "selection_fallback": False,
                "far_candidates": len(selected_by_id),
                "near_candidates": len(selected_by_id),
                "filtered_out": max(0, len(people_by_id) - len(selected_by_id)),
                "foot_x": float(fpx),
                "foot_y": float(fpy),
                "far_poly": "[]",
                "near_poly": "[]",
            })

        if debug_log:
            print(
                f"[players][frame={frame_idx}] chosen={chosen_ids} "
                f"visible={list(selected_by_id.keys())} far_id={far_id if far_id is not None else -1} "
                f"near_id={near_id if near_id is not None else -1}"
            )

    return pd.DataFrame(rows, columns=columns)
