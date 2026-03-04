from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _angle_change_deg(v_before: np.ndarray, v_after: np.ndarray) -> float:
    norm_before = float(np.linalg.norm(v_before))
    norm_after = float(np.linalg.norm(v_after))
    if norm_before <= 1e-9 or norm_after <= 1e-9:
        return 0.0

    cosang = float(np.dot(v_before, v_after) / (norm_before * norm_after))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _distance_to_player(ball_x: float, ball_y: float, player_row: pd.Series) -> float:
    points: list[tuple[float, float]] = []

    wrist_pairs = [
        (player_row.get("wrist_l_x"), player_row.get("wrist_l_y")),
        (player_row.get("wrist_r_x"), player_row.get("wrist_r_y")),
    ]
    for wx, wy in wrist_pairs:
        if pd.notna(wx) and pd.notna(wy):
            points.append((float(wx), float(wy)))

    if not points:
        if all(pd.notna(player_row.get(col)) for col in ["x1", "y1", "x2", "y2"]):
            cx = 0.5 * (float(player_row["x1"]) + float(player_row["x2"]))
            cy = 0.5 * (float(player_row["y1"]) + float(player_row["y2"]))
            points.append((cx, cy))

    if not points:
        return math.inf

    return min(math.hypot(ball_x - px, ball_y - py) for px, py in points)


def detect_hits(
    df_ball: pd.DataFrame,
    df_players: pd.DataFrame,
    min_frames_between: int = 6,
    dist_px: float = 80,
    angle_change_deg: float = 60,
    score_threshold: float = 0.4,
) -> pd.DataFrame:
    required_ball = {"frame", "cx", "cy"}
    missing_ball = required_ball - set(df_ball.columns)
    if missing_ball:
        raise ValueError(f"Missing required ball columns: {sorted(missing_ball)}")

    out_cols = ["frame_hit", "player", "cx", "cy", "hit_score"]
    if df_ball.empty or df_players.empty:
        return pd.DataFrame(columns=out_cols)

    ball = df_ball[["frame", "cx", "cy"]].copy()
    ball["frame"] = pd.to_numeric(ball["frame"], errors="coerce")
    ball["cx"] = pd.to_numeric(ball["cx"], errors="coerce")
    ball["cy"] = pd.to_numeric(ball["cy"], errors="coerce")
    ball = ball.dropna(subset=["frame", "cx", "cy"]).sort_values("frame").reset_index(drop=True)

    if len(ball) < 3:
        return pd.DataFrame(columns=out_cols)

    players = df_players.copy()
    players["frame"] = pd.to_numeric(players["frame"], errors="coerce")
    players = players.dropna(subset=["frame", "player"]).copy()

    candidates: list[dict[str, float | int | str]] = []

    for idx in range(1, len(ball) - 1):
        prev_row = ball.iloc[idx - 1]
        curr_row = ball.iloc[idx]
        next_row = ball.iloc[idx + 1]

        frame_t = int(curr_row["frame"])
        v_before = np.array([curr_row["cx"] - prev_row["cx"], curr_row["cy"] - prev_row["cy"]], dtype=np.float64)
        v_after = np.array([next_row["cx"] - curr_row["cx"], next_row["cy"] - curr_row["cy"]], dtype=np.float64)

        angle_deg = _angle_change_deg(v_before, v_after)

        players_t = players[players["frame"] == frame_t]
        if players_t.empty:
            continue

        ball_x = float(curr_row["cx"])
        ball_y = float(curr_row["cy"])

        best_player = None
        best_distance = math.inf
        for _, player_row in players_t.iterrows():
            distance = _distance_to_player(ball_x, ball_y, player_row)
            if distance < best_distance:
                best_distance = distance
                best_player = str(player_row["player"])

        if best_player is None:
            continue

        if best_distance < float(dist_px) and angle_deg > float(angle_change_deg):
            score = _clamp01(angle_deg / 180.0) * 0.7 + _clamp01((float(dist_px) - best_distance) / float(dist_px)) * 0.3
            if score >= float(score_threshold):
                candidates.append(
                    {
                        "frame_hit": frame_t,
                        "player": best_player,
                        "cx": ball_x,
                        "cy": ball_y,
                        "hit_score": float(score),
                    }
                )

    if not candidates:
        return pd.DataFrame(columns=out_cols)

    cdf = pd.DataFrame(candidates)
    cdf = cdf.sort_values("hit_score", ascending=False).reset_index(drop=True)

    selected_idx: list[int] = []
    selected_frames: list[int] = []
    for row in cdf.itertuples(index=True):
        frame_hit = int(row.frame_hit)
        if not selected_frames or min(abs(frame_hit - f) for f in selected_frames) >= int(min_frames_between):
            selected_idx.append(int(row.Index))
            selected_frames.append(frame_hit)

    if not selected_idx:
        return pd.DataFrame(columns=out_cols)

    result = cdf.loc[selected_idx, out_cols].sort_values("hit_score", ascending=False).reset_index(drop=True)
    return result
