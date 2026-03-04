#!/usr/bin/env python3
"""Render a single debug frame with near/far players, wrists and ball position.

Example:
python scripts/debug_frame.py --video path/to/input.mp4 --frame 58 --outdir outputs --players-csv outputs/players.csv --ball-csv outputs/detecciones_interpoladas.csv --assert-far-not-sideline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug a specific frame overlay (players + ball).")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--frame", type=int, required=True, help="1-based frame number to inspect")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--players-csv", type=str, default=None, help="players.csv path (near/far rows)")
    parser.add_argument("--ball-csv", type=str, default=None, help="ball CSV with frame,cx,cy (e.g. detecciones_interpoladas.csv)")
    parser.add_argument(
        "--assert-far-not-sideline",
        action="store_true",
        help="Fail if far player center_x < 0.2W for this frame",
    )
    return parser.parse_args()


def _load_frame(video_path: Path, frame_number: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_number}")
    return frame


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() / "debug"
    outdir.mkdir(parents=True, exist_ok=True)

    frame = _load_frame(video_path, int(args.frame))
    H, W = frame.shape[:2]

    if args.players_csv:
        df_players = pd.read_csv(args.players_csv)
        rows = df_players[df_players["frame"].astype(int) == int(args.frame)].copy()

        colors = {"near": (255, 0, 255), "far": (0, 255, 255)}
        for _, row in rows.iterrows():
            player = str(row["player"])
            if player not in colors:
                continue
            x1, y1, x2, y2 = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            cx = 0.5 * (float(row["x1"]) + float(row["x2"]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[player], 2)
            cv2.putText(frame, f"{player} conf={float(row.get('conf', np.nan)):.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[player], 2)

            for px, py in [(row.get("wrist_l_x", np.nan), row.get("wrist_l_y", np.nan)), (row.get("wrist_r_x", np.nan), row.get("wrist_r_y", np.nan))]:
                if pd.notna(px) and pd.notna(py):
                    cv2.circle(frame, (int(px), int(py)), 4, colors[player], -1)

            if args.assert_far_not_sideline and player == "far" and cx < 0.2 * W:
                raise AssertionError(f"Far player looks like sideline selection: center_x={cx:.1f} < {0.2 * W:.1f}")

    if args.ball_csv:
        df_ball = pd.read_csv(args.ball_csv)
        if {"frame", "cx", "cy"}.issubset(df_ball.columns):
            row = df_ball[df_ball["frame"].astype(int) == int(args.frame)]
            if not row.empty:
                cx, cy = int(float(row.iloc[0]["cx"])), int(float(row.iloc[0]["cy"]))
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"ball ({cx},{cy})", (cx + 8, max(20, cy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    outpath = outdir / f"frame_{int(args.frame)}_debug.png"
    cv2.imwrite(str(outpath), frame)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
