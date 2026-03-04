from __future__ import annotations

import numpy as np
import pandas as pd


def _clamp01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def detect_bounces(
    df: pd.DataFrame,
    fps: float,
    smooth_window: int = 7,
    min_frames_between: int = 10,
    dy_threshold_px: float = 3.0,
    score_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Detecta botes usando SOLO señal en píxeles (cx, cy) del df interpolado.

    Requisitos:
    - df tiene columnas: frame, cx, cy
    - NO modifica df in-place
    - Devuelve un DataFrame con columnas:
        frame_bounce, cx, cy, bounce_score
      ordenado por bounce_score desc, con todos los candidatos (no solo el mejor).
    """
    _ = fps  # reservado para evolución futura del detector

    required_cols = {"frame", "cx", "cy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out_cols = ["frame_bounce", "cx", "cy", "bounce_score"]
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    series = df[["frame", "cx", "cy"]].copy()
    series["cy"] = pd.to_numeric(series["cy"], errors="coerce")
    series["cx"] = pd.to_numeric(series["cx"], errors="coerce")
    series["frame"] = pd.to_numeric(series["frame"], errors="coerce")
    series = series.dropna(subset=["frame", "cx", "cy"]).reset_index(drop=True)

    if len(series) < 3:
        return pd.DataFrame(columns=out_cols)

    smooth_window = max(3, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1

    cy_smooth = (
        series["cy"].rolling(window=smooth_window, center=True, min_periods=1).mean().ffill().bfill()
    )

    dy = cy_smooth.diff().to_numpy(dtype=np.float64)
    ddy = np.diff(dy)

    dy_prev = dy[:-1]
    dy_curr = dy[1:]
    sign_flip = (dy_prev > float(dy_threshold_px)) & (dy_curr < -float(dy_threshold_px))

    if not np.any(sign_flip):
        return pd.DataFrame(columns=out_cols)

    abs_ddy = np.abs(ddy)
    delta_dy = np.abs(dy_prev - dy_curr)

    valid_abs_ddy = abs_ddy[np.isfinite(abs_ddy)]
    valid_delta = delta_dy[np.isfinite(delta_dy)]

    p95_abs_ddy = np.percentile(valid_abs_ddy, 95) if valid_abs_ddy.size else 0.0
    p95_delta_dy = np.percentile(valid_delta, 95) if valid_delta.size else 0.0

    if p95_abs_ddy <= 0:
        p95_abs_ddy = 1.0
    if p95_delta_dy <= 0:
        p95_delta_dy = 1.0

    s1 = _clamp01(abs_ddy / p95_abs_ddy)
    s2 = _clamp01(delta_dy / p95_delta_dy)
    score = 0.7 * s1 + 0.3 * s2

    candidate_idx = np.where(sign_flip)[0] + 1  # índice temporal t para dy[t]
    candidate_scores = score[candidate_idx - 1]

    candidate_df = pd.DataFrame(
        {
            "row_idx": candidate_idx,
            "frame_bounce": series.loc[candidate_idx, "frame"].to_numpy(dtype=np.int64),
            "cx": series.loc[candidate_idx, "cx"].to_numpy(dtype=np.float64),
            "cy": series.loc[candidate_idx, "cy"].to_numpy(dtype=np.float64),
            "bounce_score": candidate_scores.astype(np.float64),
        }
    )

    candidate_df = candidate_df[candidate_df["bounce_score"] >= float(score_threshold)].copy()
    if candidate_df.empty:
        return pd.DataFrame(columns=out_cols)

    candidate_df = candidate_df.sort_values("bounce_score", ascending=False).reset_index(drop=True)

    selected_rows: list[int] = []
    selected_frames: list[int] = []
    for row in candidate_df.itertuples(index=False):
        frame_val = int(row.frame_bounce)
        if not selected_frames or min(abs(frame_val - f) for f in selected_frames) >= int(min_frames_between):
            selected_rows.append(int(row.row_idx))
            selected_frames.append(frame_val)

    if not selected_rows:
        return pd.DataFrame(columns=out_cols)

    selected = candidate_df[candidate_df["row_idx"].isin(selected_rows)].copy()
    selected = selected[out_cols].sort_values("bounce_score", ascending=False).reset_index(drop=True)
    return selected
