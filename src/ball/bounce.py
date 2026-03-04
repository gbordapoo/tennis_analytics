
from __future__ import annotations

import numpy as np
import pandas as pd


def _clamp01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _min_distance_to_frames(frames: np.ndarray, ref_frames: np.ndarray) -> np.ndarray:
    if frames.size == 0 or ref_frames.size == 0:
        return np.full(frames.shape, np.inf, dtype=np.float64)
    ref_frames = np.sort(ref_frames.astype(np.int64))
    idx = np.searchsorted(ref_frames, frames, side="left")
    left_idx = np.clip(idx - 1, 0, len(ref_frames) - 1)
    right_idx = np.clip(idx, 0, len(ref_frames) - 1)
    left_dist = np.abs(frames - ref_frames[left_idx])
    right_dist = np.abs(frames - ref_frames[right_idx])
    return np.minimum(left_dist, right_dist).astype(np.float64)


def detect_bounces(
    df: pd.DataFrame,
    fps: float,
    smooth_window: int = 5,
    min_frames_between: int = 10,
    dy_threshold_px: float = 1.0,
    score_threshold: float = 0.2,
    exclude_frames: set[int] | None = None,
    hit_frames: list[int] | None = None,
    exclude_post_hit: int = 4,
    exclude_pre_hit: int = 0,
    speed_min: float = 2.0,
    min_drop_px: float = 6.0,
    local_window: int = 3,
    max_gap_to_raw: int = 20,
    raw_detected_frames: list[int] | None = None,
) -> pd.DataFrame:
    """Detect bounce candidates from interpolated trajectory with reliability filters.

    Returns all scored candidates (not just threshold-passing rows) so debugging can
    tune thresholds post-run. Rows include extra debug columns used by overlays.
    """
    _ = fps

    required_cols = {"frame", "cx", "cy", "vy", "speed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out_cols = [
        "frame_bounce",
        "cx",
        "cy",
        "bounce_score",
        "passes_threshold",
        "selected",
        "vy_pre",
        "vy_post",
        "cy_drop",
        "cy_rise",
        "is_extrapolated",
        "speed",
    ]

    if df.empty:
        return pd.DataFrame(columns=out_cols)

    series = df.copy()
    for col in ["frame", "cx", "cy", "vy", "speed"]:
        series[col] = pd.to_numeric(series[col], errors="coerce")
    if "extrapolated" not in series.columns:
        series["extrapolated"] = False

    series = series.dropna(subset=["frame", "cx", "cy", "vy", "speed"]).sort_values("frame").reset_index(drop=True)
    if len(series) < 3:
        return pd.DataFrame(columns=out_cols)

    smooth_window = max(3, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1

    local_window = max(2, int(local_window))
    cy_smooth = series["cy"].rolling(window=smooth_window, center=True, min_periods=1).mean().ffill().bfill()
    vy_smooth = series["vy"].rolling(window=smooth_window, center=True, min_periods=1).mean().ffill().bfill()

    is_extrap = series["extrapolated"].fillna(False).astype(bool).to_numpy()
    speed = series["speed"].to_numpy(dtype=np.float64)
    frame_vals = series["frame"].to_numpy(dtype=np.int64)

    reliable_mask = (~is_extrap) & np.isfinite(speed) & (speed >= float(speed_min))
    if raw_detected_frames:
        raw_frames = np.array(sorted({int(f) for f in raw_detected_frames}), dtype=np.int64)
        dist_raw = _min_distance_to_frames(frame_vals, raw_frames)
        reliable_mask &= dist_raw <= int(max_gap_to_raw)

    if int(reliable_mask.sum()) < 6:
        print("⚠️ Bounce detection skipped: too few reliable trajectory frames after filtering.")
        return pd.DataFrame(columns=out_cols)

    cy_vals = cy_smooth.to_numpy(dtype=np.float64)
    vy_vals = vy_smooth.to_numpy(dtype=np.float64)

    candidate_rows: list[dict[str, float | int | bool]] = []
    for idx in range(local_window, len(series) - local_window):
        if not reliable_mask[idx]:
            continue

        local = cy_vals[idx - local_window : idx + local_window + 1]
        if cy_vals[idx] > np.nanmin(local):
            continue

        pre_slice = slice(idx - local_window, idx)
        post_slice = slice(idx + 1, idx + local_window + 1)

        vy_pre = float(np.nanmean(vy_vals[pre_slice]))
        vy_post = float(np.nanmean(vy_vals[post_slice]))
        if not (np.isfinite(vy_pre) and np.isfinite(vy_post)):
            continue
        if not (vy_pre < -abs(float(dy_threshold_px)) and vy_post > abs(float(dy_threshold_px))):
            continue

        pre_max = float(np.nanmax(cy_vals[pre_slice]))
        post_max = float(np.nanmax(cy_vals[post_slice]))
        cy_drop = pre_max - float(cy_vals[idx])
        cy_rise = post_max - float(cy_vals[idx])

        if cy_drop < float(min_drop_px) or cy_rise < float(min_drop_px):
            continue

        score_drop = min(cy_drop / max(float(min_drop_px), 1e-6), 3.0) / 3.0
        score_rise = min(cy_rise / max(float(min_drop_px), 1e-6), 3.0) / 3.0
        score_sign = _clamp01(np.array([(-vy_pre + vy_post) / max(2.0 * abs(float(dy_threshold_px)), 1e-6)]))[0]
        score_speed = _clamp01(np.array([speed[idx] / max(float(speed_min), 1e-6)]))[0]
        bounce_score = float(0.35 * score_drop + 0.35 * score_rise + 0.2 * score_sign + 0.1 * score_speed)

        candidate_rows.append(
            {
                "row_idx": idx,
                "frame_bounce": int(frame_vals[idx]),
                "cx": float(series.loc[idx, "cx"]),
                "cy": float(series.loc[idx, "cy"]),
                "bounce_score": bounce_score,
                "vy_pre": vy_pre,
                "vy_post": vy_post,
                "cy_drop": cy_drop,
                "cy_rise": cy_rise,
                "is_extrapolated": bool(is_extrap[idx]),
                "speed": float(speed[idx]),
            }
        )

    if not candidate_rows:
        return pd.DataFrame(columns=out_cols)

    candidate_df = pd.DataFrame(candidate_rows)

    if exclude_frames:
        candidate_df = candidate_df[~candidate_df["frame_bounce"].astype(int).isin(exclude_frames)].copy()

    if hit_frames:
        exclude_ranges = [
            (int(hit) - int(exclude_pre_hit), int(hit) + int(exclude_post_hit))
            for hit in hit_frames
        ]
        candidate_df = candidate_df[
            ~candidate_df["frame_bounce"].astype(int).apply(
                lambda frame: any(start <= frame <= end for start, end in exclude_ranges)
            )
        ].copy()

    if candidate_df.empty:
        return pd.DataFrame(columns=out_cols)

    candidate_df = candidate_df.sort_values("bounce_score", ascending=False).reset_index(drop=True)
    candidate_df["passes_threshold"] = candidate_df["bounce_score"] >= float(score_threshold)
    candidate_df["selected"] = False

    selected_frames: list[int] = []
    for ridx, row in candidate_df.iterrows():
        frame_val = int(row["frame_bounce"])
        if not selected_frames or min(abs(frame_val - f) for f in selected_frames) >= int(min_frames_between):
            if bool(row["passes_threshold"]):
                candidate_df.loc[ridx, "selected"] = True
                selected_frames.append(frame_val)

    return candidate_df[out_cols].reset_index(drop=True)
