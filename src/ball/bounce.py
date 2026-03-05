from __future__ import annotations

import numpy as np
import pandas as pd


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


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
    top_k: int = 8,
) -> pd.DataFrame:
    """Detect bounce candidates using robust vertical flip/impact signals.

    The detector keeps candidates for debugging (top-K sorted by score) even when no
    row passes score threshold.
    """
    _ = fps
    _ = smooth_window
    _ = max_gap_to_raw
    _ = raw_detected_frames

    required_cols = {"frame", "cx", "cy", "speed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out_cols = [
        "frame_bounce",
        "cx",
        "cy",
        "bounce_score",
        "type",
        "vy_prev",
        "vy",
        "ay",
        "speed",
        "cy_zone",
        "extrapolated",
        "passes_threshold",
        "selected",
    ]

    if df.empty:
        return pd.DataFrame(columns=out_cols)

    series = df.copy()
    series["frame"] = pd.to_numeric(series["frame"], errors="coerce")
    series["cx"] = pd.to_numeric(series["cx"], errors="coerce")
    series["cy"] = pd.to_numeric(series["cy"], errors="coerce")
    series["speed"] = pd.to_numeric(series["speed"], errors="coerce")
    if "vy" in series.columns:
        series["vy"] = pd.to_numeric(series["vy"], errors="coerce")
    else:
        series["vy"] = np.nan

    if "extrapolated" not in series.columns:
        series["extrapolated"] = False

    series = series.dropna(subset=["frame", "cx", "cy", "speed"]).sort_values("frame")
    series = series.drop_duplicates(subset=["frame"], keep="first").reset_index(drop=True)
    if len(series) < 4:
        return pd.DataFrame(columns=out_cols)

    # Robust derivatives from cy. Prefer this over raw vy to avoid inconsistent values.
    vy_from_cy = series["cy"].diff().fillna(0.0)
    vy = vy_from_cy.where(vy_from_cy.notna(), series["vy"]).astype(float)
    ay = vy.diff().fillna(0.0).astype(float)

    series["vy_calc"] = vy
    series["ay_calc"] = ay

    extrap = series["extrapolated"].fillna(False).astype(bool)
    speed = series["speed"].astype(float)
    reliable_mask = (~extrap) & np.isfinite(speed) & (speed >= float(speed_min))

    reliable_cy = series.loc[reliable_mask, "cy"].astype(float)
    if reliable_cy.empty:
        return pd.DataFrame(columns=out_cols)

    p30 = float(np.percentile(reliable_cy, 30))
    p70 = float(np.percentile(reliable_cy, 70))

    def _zone(cy: float) -> str:
        if cy < p30:
            return "far"
        if cy > p70:
            return "near"
        return "mid"

    eps = max(1.5, float(dy_threshold_px))
    ay_thr = 12.0
    local_window = max(2, int(local_window))

    candidate_rows: list[dict[str, float | int | str | bool]] = []
    for idx in range(1, len(series) - 1):
        if not bool(reliable_mask.iloc[idx]):
            continue

        vy_prev = float(vy.iloc[idx - 1])
        vy_now = float(vy.iloc[idx])
        ay_now = float(ay.iloc[idx])
        cy_now = float(series.loc[idx, "cy"])
        frame_now = int(series.loc[idx, "frame"])

        flip = vy_prev > eps and vy_now < -eps
        impact = ay_now < -ay_thr and abs(vy_now) < abs(vy_prev) * 0.4
        if not (flip or impact):
            continue

        left = max(0, idx - local_window)
        right = min(len(series) - 1, idx + local_window)
        pre_max = float(series.loc[left:idx, "cy"].max())
        post_max = float(series.loc[idx:right, "cy"].max())
        cy_drop = pre_max - cy_now
        cy_rise = post_max - cy_now

        cy_zone = _zone(cy_now)
        zone_bonus = 0.08 if cy_zone in {"near", "far"} else 0.0
        drop_bonus = 0.15 * _clamp01(min(cy_drop, cy_rise) / max(float(min_drop_px), 1e-6))
        speed_score = _clamp01(float(series.loc[idx, "speed"]) / max(float(speed_min), 1e-6))
        ay_score = _clamp01((-ay_now) / ay_thr)

        flip_score = 0.55 if flip else 0.0
        impact_score = 0.45 if impact else 0.0
        bounce_score = float(min(1.5, flip_score + impact_score + 0.2 * ay_score + 0.1 * speed_score + zone_bonus + drop_bonus))

        candidate_rows.append(
            {
                "frame_bounce": frame_now,
                "cx": float(series.loc[idx, "cx"]),
                "cy": cy_now,
                "bounce_score": bounce_score,
                "type": "flip" if flip else "impact",
                "vy_prev": vy_prev,
                "vy": vy_now,
                "ay": ay_now,
                "speed": float(series.loc[idx, "speed"]),
                "cy_zone": cy_zone,
                "extrapolated": bool(extrap.iloc[idx]),
            }
        )

    if not candidate_rows:
        return pd.DataFrame(columns=out_cols)

    candidate_df = pd.DataFrame(candidate_rows)

    if exclude_frames:
        candidate_df = candidate_df[~candidate_df["frame_bounce"].astype(int).isin(exclude_frames)].copy()

    if hit_frames:
        hit_ranges = [(int(h) - int(exclude_pre_hit), int(h) + int(exclude_post_hit)) for h in hit_frames]
        candidate_df = candidate_df[
            ~candidate_df["frame_bounce"].astype(int).apply(lambda f: any(lo <= f <= hi for lo, hi in hit_ranges))
        ].copy()

    if candidate_df.empty:
        return pd.DataFrame(columns=out_cols)

    candidate_df = candidate_df.sort_values("bounce_score", ascending=False).drop_duplicates(subset=["frame_bounce"], keep="first")
    candidate_df = candidate_df.head(max(1, int(top_k))).reset_index(drop=True)

    candidate_df["passes_threshold"] = candidate_df["bounce_score"] >= float(score_threshold)
    candidate_df["selected"] = False

    selected_frames: list[int] = []
    for ridx, row in candidate_df.iterrows():
        frame_val = int(row["frame_bounce"])
        if row["passes_threshold"] and (not selected_frames or min(abs(frame_val - sf) for sf in selected_frames) >= int(min_frames_between)):
            candidate_df.loc[ridx, "selected"] = True
            selected_frames.append(frame_val)

    return candidate_df[out_cols]
