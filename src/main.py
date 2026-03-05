from __future__ import annotations

import argparse
from pathlib import Path




def _resolve_path(path_str: str, project_root: Path, script_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()

    project_candidate = (project_root / path).resolve()
    if project_candidate.exists():
        return project_candidate

    script_candidate = (script_dir / path).resolve()
    if script_candidate.exists():
        return script_candidate

    return project_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular tennis analytics pipeline")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--ball-model", type=str, default="models/yolo5_last.pt", help="Path to YOLOv5 tennis-ball model")
    parser.add_argument("--court-model", type=str, default="models/keypoints_model.pth", help="Path to court keypoints model")
    parser.add_argument("--player-model", type=str, default="models/yolov8n.pt", help="Path to YOLOv8 player model")
    parser.add_argument("--output", type=str, default="outputs/run1.mp4", help="Output video path")

    # Compatibility aliases for previous CLI
    parser.add_argument("--model", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--outdir", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import cv2

    from court.court_geometry import CourtGeometry
    from court.keypoint_detector import CourtKeypointDetector
    from detection.ball_detector import BallDetector
    from detection.player_detector import PlayerDetector
    from tracking.object_tracker import SimpleTracker
    from viz.render import draw_ball, draw_keypoints, draw_players

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    ball_model = args.model or args.ball_model
    output = args.output
    if args.outdir:
        output = str(Path(args.outdir) / "output_ultralytics.mp4")

    video_path = _resolve_path(args.video, project_root, script_dir)
    ball_model_path = _resolve_path(ball_model, project_root, script_dir)
    court_model_path = _resolve_path(args.court_model, project_root, script_dir)
    player_model_path = _resolve_path(args.player_model, project_root, script_dir)
    output_path = _resolve_path(output, project_root, script_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    player_detector = PlayerDetector(str(player_model_path))
    ball_detector = BallDetector(str(ball_model_path))
    court_detector = CourtKeypointDetector(str(court_model_path))
    tracker = SimpleTracker()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))

    frame_id = 0
    geometry = None
    court_kp = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        players = player_detector.detect(frame)
        balls = ball_detector.detect(frame)
        _ = tracker.update(balls)

        if frame_id == 0:
            court_kp = court_detector.predict(frame, scale_to_frame=True)
            geometry = CourtGeometry(court_kp)

            first_three = court_kp[:3].tolist()
            in_bounds = all(0 <= int(x) < width and 0 <= int(y) < height for x, y in first_three)
            print(f"Frame 0 keypoints (first 3, scaled): {first_three}; within_bounds={in_bounds}")

        near_player = None
        far_player = None
        if geometry is not None:
            filtered = geometry.filter_players(players)
            near_player, far_player = geometry.assign_players(filtered)

        draw_players(frame, near_player, far_player)
        draw_ball(frame, balls)
        if court_kp is not None:
            draw_keypoints(frame, court_kp)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"✅ Output saved to: {output_path}")


if __name__ == "__main__":
    main()
