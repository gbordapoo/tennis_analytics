from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ultralytics import YOLO

LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10


def ensure_pose_model(weights_path: Path) -> Path:
    """Ensure pose weights exist at a fixed local path."""
    target = Path(weights_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        return target

    try:
        model = YOLO("yolov8n-pose.pt")
        source = Path(model.ckpt_path)
        if not source.exists():
            raise FileNotFoundError(f"Ultralytics checkpoint not found at {source}")
        target.write_bytes(source.read_bytes())
    except Exception as exc:
        raise RuntimeError(
            "Failed to download pose model automatically. "
            f"Download 'yolov8n-pose.pt' manually and place it at {target}."
        ) from exc

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
    weights_path: str | Path = "yolov8n-pose.pt",
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

    model = YOLO(str(weights_path))
    rows: list[dict[str, Any]] = []

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

        top_people = sorted(people, key=lambda item: item["conf"], reverse=True)[:2]
        top_people = sorted(top_people, key=lambda item: item["y2"], reverse=True)

        for idx, person in enumerate(top_people):
            label = "near" if idx == 0 else "far"
            rows.append(
                {
                    "frame": frame_idx,
                    "player": label,
                    **person,
                }
            )

    return pd.DataFrame(rows, columns=columns)
