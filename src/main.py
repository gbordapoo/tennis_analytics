from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from analytics.assign_players import assign_near_far_players
from court.court_detector import TennisCourtDetector
from detection.ball import BallDetector
from detection.players import PlayerDetector
from tracking.ball_track import SimpleBallTracker, SimpleCentroidTracker
from viz.render import render_frame


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(path_str: str, project_root: Path, script_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    p1 = (project_root / path).resolve()
    if p1.exists():
        return p1
    p2 = (script_dir / path).resolve()
    if p2.exists():
        return p2
    return p1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modular tennis analytics pipeline")
    p.add_argument("--video", required=True)
    p.add_argument("--ball-model", default="models/yolo5_last.pt")
    p.add_argument("--court-model", default="models/model_tennis_court_det.pt")
    p.add_argument("--player-model", default="models/yolov8n.pt")
    p.add_argument("--output", default="outputs/run1.mp4")
    p.add_argument("--player-every", type=int, default=2)
    p.add_argument("--debug-court", default=None)
    p.add_argument("--debug-frame", type=int, default=0)
    p.add_argument("--court-refine-lines", type=_parse_bool, default=True)
    p.add_argument("--court-refine-homography", type=_parse_bool, default=True)
    p.add_argument("--court-crop-size", type=int, default=40)
    p.add_argument("--court-max-shift-px", type=float, default=35)

    # compatibility aliases
    p.add_argument("--model", default=None, help=argparse.SUPPRESS)
    p.add_argument("--outdir", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    ball_model = args.model or args.ball_model
    output = args.output if args.outdir is None else str(Path(args.outdir) / "output_ultralytics.mp4")

    video_path = _resolve_path(args.video, root, script_dir)
    ball_model_path = _resolve_path(ball_model, root, script_dir)
    court_model_path = _resolve_path(args.court_model, root, script_dir)
    player_model_path = _resolve_path(args.player_model, root, script_dir)
    output_path = _resolve_path(output, root, script_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = _choose_device()
    court_detector = TennisCourtDetector(
        str(court_model_path),
        device=device,
        refine_lines=args.court_refine_lines,
        refine_homography=args.court_refine_homography,
        crop_size=args.court_crop_size,
        max_shift_px=args.court_max_shift_px,
    )
    ball_detector = BallDetector(str(ball_model_path))
    player_detector = PlayerDetector(str(player_model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))

    ball_tracker = SimpleBallTracker()
    player_tracker = SimpleCentroidTracker()

    cached_players = []
    court_keypoints = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx == 0:
            court_keypoints = court_detector.detect(frame)

        kps = None
        if court_keypoints is not None and all(x is not None and y is not None for x, y in court_keypoints):
            kps = court_keypoints

        if args.debug_court and frame_idx == args.debug_frame:
            debug_img = render_frame(frame, kps, None, None, None)
            debug_path = _resolve_path(args.debug_court, root, script_dir)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), debug_img)

        if frame_idx % max(1, args.player_every) == 0:
            cached_players = player_detector.detect(frame)
        player_tracker.update(cached_players)

        near_player, far_player = assign_near_far_players(cached_players, kps)

        balls = ball_detector.detect(frame)
        ball_center = ball_tracker.update(balls)

        out_frame = render_frame(frame, kps, near_player, far_player, ball_center)
        if court_keypoints is not None:
            out_frame = court_detector.draw(out_frame, court_keypoints)
        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved output video: {output_path}")


if __name__ == "__main__":
    main()
