from __future__ import annotations

import argparse
from pathlib import Path

import torch


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
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--ball-model", type=str, default="models/yolo5_last.pt")
    parser.add_argument("--court-model", type=str, default="models/keypoints_model.pth")
    parser.add_argument("--player-model", type=str, default="models/yolov8n.pt")
    parser.add_argument("--output", type=str, default="outputs/run1.mp4")

    parser.add_argument("--court-frames", type=int, default=30)
    parser.add_argument("--court-refresh-every", type=int, default=0)
    parser.add_argument("--court-ema-alpha", type=float, default=0.2)
    parser.add_argument("--court-refine-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-players", action="store_true")
    parser.add_argument("--no-court", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--model", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--outdir", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _doctor_check(video_path: Path, ball_model: Path, court_model: Path, player_model: Path, no_players: bool, no_court: bool, device: str) -> None:
    required = [video_path, ball_model]
    if not no_court:
        required.append(court_model)
    if not no_players:
        required.append(player_model)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required local files:\n" + "\n".join(missing))
    print(f"[doctor] device={device}")


def main() -> None:
    args = parse_args()

    import cv2

    from analytics.assign import assign_near_far_players
    from court.geometry import refine_keypoints
    from court.infer import infer_output_domain, predict_keypoints, preprocess
    from court.model import load_keypoints_model
    from court.stabilize import ema_update, estimate_stable_keypoints
    from detection.ball import BallDetector
    from detection.players import PlayerDetector
    from tracking.ball_track import SimpleBallTracker
    from viz.draw import draw_ball, draw_keypoints, draw_players

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

    device = _choose_device()
    _doctor_check(video_path, ball_model_path, court_model_path, player_model_path, args.no_players, args.no_court, device)

    ball_detector = BallDetector(str(ball_model_path))
    player_detector = None if args.no_players else PlayerDetector(str(player_model_path))
    tracker = SimpleBallTracker()

    court_model = None
    stable_kpts = None
    if not args.no_court:
        court_model = load_keypoints_model(str(court_model_path), device=device, num_keypoints=14)
        stable_kpts = estimate_stable_keypoints(str(video_path), court_model, device, num_frames=args.court_frames)
        if args.court_refine_geometry:
            stable_kpts = refine_keypoints(stable_kpts)

        cap_probe = cv2.VideoCapture(str(video_path))
        ok, probe_frame = cap_probe.read()
        cap_probe.release()
        if ok:
            raw = court_model(preprocess(probe_frame).to(device)).detach().cpu().numpy().reshape(-1, 2)
            print(f"[doctor] court output domain heuristic={infer_output_domain(raw)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        balls = ball_detector.detect(frame)
        _ = tracker.update(balls)

        if (not args.no_court) and args.court_refresh_every > 0 and frame_id > 0 and frame_id % args.court_refresh_every == 0:
            raw_kpts = predict_keypoints(court_model, frame, device)
            if args.court_refine_geometry:
                raw_kpts = refine_keypoints(raw_kpts)
            stable_kpts = ema_update(stable_kpts, raw_kpts, alpha=args.court_ema_alpha)

        near_player, far_player = None, None
        if player_detector is not None:
            players = player_detector.detect(frame)
            if stable_kpts is not None:
                near_player, far_player = assign_near_far_players(players, stable_kpts)

        draw_ball(frame, balls)
        draw_players(frame, near_player, far_player)
        if stable_kpts is not None:
            draw_keypoints(frame, stable_kpts)

        if args.debug and frame_id == 0:
            debug_path = output_path.parent / "debug_first_frame.png"
            cv2.imwrite(str(debug_path), frame)
            print(f"[debug] saved {debug_path}")

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"✅ Output saved to: {output_path}")


if __name__ == "__main__":
    main()
