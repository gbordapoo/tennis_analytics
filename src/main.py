from __future__ import annotations

import argparse
import json
from pathlib import Path

from ball.bounce import detect_bounces
from ball.detect import load_model, run_detection
from ball.hit import detect_hits
from ball.track import interpolar_detecciones
from court.auto_calibrate import draw_court_overlay, run_static_auto_calibration
from court.keypoints import load_keypoints_model, predict_court_keypoints
from court.homography import (
    apply_homography,
    compute_homography,
    load_manual_calibration,
    project_points_to_meters,
)
from player.pose import detect_players, ensure_pose_model
from viz.render import render_video, save_direction_plots


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


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




def _build_player_selection_cfg(args: argparse.Namespace, project_root: Path, script_dir: Path) -> dict[str, float | int]:
    cfg: dict[str, float | int] = {
        "player_xgate_left": args.player_xgate_left,
        "player_xgate_right": args.player_xgate_right,
        "player_min_conf": args.player_min_conf,
        "player_halfcourt_margin_px": args.player_halfcourt_margin_px,
    }

    if args.calibration is None:
        return cfg

    calibration_path = _resolve_path(args.calibration, project_root, script_dir)

    if not calibration_path.exists():
        print(f"⚠️ Calibration file for player gate not found: {calibration_path}")
        return cfg

    try:
        pixel_points, _ = load_manual_calibration(calibration_path)
        cfg["calibration_far_left_x"] = float(pixel_points[2][0])
        cfg["calibration_far_right_x"] = float(pixel_points[3][0])
        cfg["calibration_pixel_points"] = pixel_points
    except Exception as exc:
        print(f"⚠️ Could not load calibration for player gate: {exc}")

    return cfg

def _filter_bounces_with_next_hit(df_bounces, df_hits, max_gap: int):
    if df_bounces.empty or df_hits is None or df_hits.empty:
        return df_bounces

    hit_frames = sorted(df_hits["frame_hit"].astype(int).tolist())
    if not hit_frames:
        return df_bounces

    keep_mask = []
    for frame_bounce in df_bounces["frame_bounce"].astype(int).tolist():
        next_hits = [hit for hit in hit_frames if hit > frame_bounce]
        if not next_hits:
            keep_mask.append(True)
            continue

        gap = next_hits[0] - frame_bounce
        keep_mask.append(1 <= gap <= int(max_gap))

    return df_bounces.loc[keep_mask].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Tennis Ball Detection + Interpolation/Extrapolation")
    parser.add_argument("--model", type=str, default="models/yolo5_last.pt", help="Path to YOLO ball detector .pt model")
    parser.add_argument("--video", type=str, default="../videos/federer_murray_trim.mp4", help="Path to input video")
    parser.add_argument("--outdir", type=str, default="../outputs", help="Output directory")
    parser.add_argument("--extrap", type=int, default=5, help="Number of frames to extrapolate backwards")
    parser.add_argument("--no-gui", action="store_true", help="Disable cv2.imshow (headless mode)")
    parser.add_argument("--auto-calibrate", action="store_true", help="Enable court homography calibration")
    parser.add_argument(
        "--calibration-mode",
        type=str,
        default="auto",
        choices=["auto", "static", "manual"],
        help="Calibration mode for homography (Phase 1: auto behaves like static)",
    )
    parser.add_argument("--calibration", type=str, default=None, help="Path to manual calibration JSON file")
    parser.add_argument("--calibration-visuals", action="store_true", help="Render calibration points/polygon on output video")
    parser.add_argument("--detect-bounces", action="store_true", help="Enable bounce detection (pixel domain)")
    parser.add_argument("--bounce-visuals", action="store_true", help="Render bounce markers on output video")
    parser.add_argument("--bounce-topk", type=int, default=3, help="Number of top bounce candidates to visualize")
    parser.add_argument("--bounce-smooth-window", type=int, default=5, help="Bounce detector smoothing window size")
    parser.add_argument(
        "--bounce-min-frames-between",
        type=int,
        default=8,
        help="Minimum frame distance between bounce candidates",
    )
    parser.add_argument(
        "--bounce-dy-threshold",
        "--bounce-dy-threshold-px",
        dest="bounce_dy_threshold",
        type=float,
        default=1.0,
        help="Minimum vertical velocity change threshold in pixels",
    )
    parser.add_argument(
        "--bounce-score-threshold",
        type=float,
        default=0.2,
        help="Minimum bounce confidence score",
    )
    parser.add_argument("--bounce-speed-min", type=float, default=2.0, help="Minimum speed (px/frame) for reliable bounce frames")
    parser.add_argument("--bounce-min-drop-px", type=float, default=6.0, help="Minimum vertical drop/rise around bounce candidate")
    parser.add_argument(
        "--bounce-exclude-post-hit",
        type=int,
        default=4,
        help="Frames to exclude after each hit when selecting bounces",
    )
    parser.add_argument(
        "--bounce-exclude-pre-hit",
        type=int,
        default=0,
        help="Frames to exclude before each hit when selecting bounces",
    )
    parser.add_argument("--bounce-out", type=str, default="bounces.csv", help="Path to bounces CSV output")
    parser.add_argument(
        "--bounce-best-out",
        type=str,
        default="bounce_best.json",
        help="Path to best bounce JSON output",
    )
    parser.add_argument("--detect-players", action="store_true", help="Enable player pose detection")
    parser.add_argument("--draw-players", dest="draw_players", action="store_true", default=True, help="Draw near/far player boxes in output video")
    parser.add_argument("--no-draw-players", dest="draw_players", action="store_false", help="Disable player box rendering")
    parser.add_argument("--pose-model", type=str, default="models/yolov8n-pose.pt", help="Path to YOLO pose model")
    parser.add_argument(
        "--player-xgate-left",
        type=float,
        default=0.18,
        help="Left screen fraction used to gate player foot positions",
    )
    parser.add_argument(
        "--player-xgate-right",
        type=float,
        default=0.82,
        help="Right screen fraction used to gate player foot positions",
    )
    parser.add_argument(
        "--player-calibration-margin-px",
        type=float,
        default=20.0,
        help="Extra horizontal margin in pixels when using calibration far-left/far-right points for player gate",
    )
    parser.add_argument("--player-min-conf", type=float, default=0.3, help="Minimum person confidence used for near/far assignment")
    parser.add_argument(
        "--player-halfcourt-margin-px",
        type=float,
        default=16.0,
        help="Margin in pixels used to expand near/far half-court polygons for player filtering",
    )
    parser.add_argument(
        "--player-geometry-visuals",
        action="store_true",
        help="Draw near/far half-court polygons and selected player foot points",
    )
    parser.add_argument("--pose-download", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--detect-hits", action="store_true", help="Enable hit detection")
    parser.add_argument("--hit-out", type=str, default="hits.csv", help="Path to hits CSV output")
    parser.add_argument("--hit-visuals", action="store_true", help="Render hit markers on output video")
    parser.add_argument(
        "--bounce-max-gap-to-next-hit",
        type=int,
        default=15,
        help="Maximum allowed frame gap between bounce and its next hit",
    )
    parser.add_argument("--debug-players", action="store_true", help="Draw and log selected near/far players")
    parser.add_argument("--debug-court", action="store_true", help="Draw detected court keypoints (14 points)")
    parser.add_argument("--debug-ball", action="store_true", help="Draw per-frame ball detection boxes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Make paths robust regardless of where you run from
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    player_selection_cfg = _build_player_selection_cfg(args, project_root, script_dir)
    player_selection_cfg["calibration_x_margin_px"] = args.player_calibration_margin_px
    player_selection_cfg["player_min_conf"] = args.player_min_conf
    player_selection_cfg["player_debug_log"] = bool(args.detect_players or args.debug_players)
    model_path = _resolve_path(args.model, project_root, script_dir)
    video_path = _resolve_path(args.video, project_root, script_dir)
    outdir = _resolve_path(args.outdir, project_root, script_dir)
    pose_model_path = _resolve_path(args.pose_model, project_root, script_dir)
    keypoints_model_path = _resolve_path("models/keypoints_model.pth", project_root, script_dir)

    outdir.mkdir(parents=True, exist_ok=True)

    _ensure_file(model_path, "Model file")
    _ensure_file(video_path, "Video file")

    output_video = outdir / "output_ultralytics.mp4"
    output_csv = outdir / "detecciones_ultralytics.csv"
    output_csv_interpolado = outdir / "detecciones_interpoladas.csv"
    output_csv_metros = outdir / "detecciones_metros.csv"
    output_angle_png = outdir / "grafico_angulo_por_frame.png"
    output_rose_png = outdir / "rosa_direcciones.png"
    output_court_overlay = outdir / "court_overlay.png"

    bounce_out_path = Path(args.bounce_out)
    output_bounces = bounce_out_path if bounce_out_path.is_absolute() else (outdir / bounce_out_path)
    bounce_best_out_path = Path(args.bounce_best_out)
    output_bounce_best = bounce_best_out_path if bounce_best_out_path.is_absolute() else (outdir / bounce_best_out_path)
    hit_out_path = Path(args.hit_out)
    output_hits = hit_out_path if hit_out_path.is_absolute() else (outdir / hit_out_path)
    output_players = outdir / "players.csv"

    model = load_model(model_path)
    frames_raw, df_detecciones, video_info = run_detection(model, video_path)

    court_kps: list[float] | None = None
    if frames_raw:
        try:
            kp_model = load_keypoints_model(keypoints_model_path)
            court_kps = predict_court_keypoints(frames_raw[0], kp_model)
            player_selection_cfg["court_keypoints_14"] = court_kps
            if args.debug_court:
                print(f"[court] keypoints_14={court_kps}")
        except Exception as exc:
            print(f"⚠️ Could not infer court keypoints: {exc}")
    df_detecciones.to_csv(output_csv, index=False)

    if df_detecciones.empty:
        print("⚠️ No se encontraron detecciones.")
        return

    df_interpolado = interpolar_detecciones(
        df_detecciones,
        total_frames=video_info.total_frames,
        extrap_frames=int(args.extrap),
    )
    df_interpolado.to_csv(output_csv_interpolado, index=False)

    df_bounces = None
    df_players = None
    df_hits = None

    if args.detect_players or args.detect_hits:
        pose_weights = ensure_pose_model(pose_model_path)
        df_players = detect_players(
            frames_raw,
            weights_path=pose_weights,
            selection_cfg=player_selection_cfg,
        )
        output_players.parent.mkdir(parents=True, exist_ok=True)
        df_players.to_csv(output_players, index=False)
        print(f"🧍 CSV jugadores: {output_players}")

    if args.detect_hits:
        if df_players is None:
            df_players = detect_players(
                frames_raw,
                weights_path=pose_weights,
                selection_cfg=player_selection_cfg,
            )
            output_players.parent.mkdir(parents=True, exist_ok=True)
            df_players.to_csv(output_players, index=False)
            print(f"🧍 CSV jugadores: {output_players}")
        df_hits = detect_hits(df_interpolado, df_players)
        output_hits.parent.mkdir(parents=True, exist_ok=True)
        df_hits.to_csv(output_hits, index=False)
        print(f"🏓 CSV golpes: {output_hits}")

    if args.detect_bounces:
        df_bounces = detect_bounces(
            df_interpolado,
            fps=float(video_info.fps),
            smooth_window=args.bounce_smooth_window,
            min_frames_between=args.bounce_min_frames_between,
            dy_threshold_px=args.bounce_dy_threshold,
            score_threshold=args.bounce_score_threshold,
            speed_min=args.bounce_speed_min,
            min_drop_px=args.bounce_min_drop_px,
            raw_detected_frames=df_detecciones["frame"].astype(int).tolist(),
            hit_frames=(
                df_hits["frame_hit"].astype(int).tolist()
                if args.detect_hits and df_hits is not None and not df_hits.empty
                else None
            ),
            exclude_post_hit=args.bounce_exclude_post_hit,
            exclude_pre_hit=args.bounce_exclude_pre_hit,
            top_k=args.bounce_topk,
        )
        if args.detect_hits and df_hits is not None and not df_hits.empty:
            df_bounces = _filter_bounces_with_next_hit(
                df_bounces,
                df_hits,
                max_gap=args.bounce_max_gap_to_next_hit,
            )

    H = None
    overlay_frame = None
    overlay_points = None
    if args.auto_calibrate:
        try:
            if args.calibration_mode == "manual":
                if args.calibration is None:
                    raise ValueError("--calibration is required when --calibration-mode manual")
                calibration_path = (
                    (script_dir / args.calibration).resolve()
                    if not Path(args.calibration).is_absolute()
                    else Path(args.calibration).resolve()
                )
                _ensure_file(calibration_path, "Calibration file")
                pixel_points, world_points = load_manual_calibration(calibration_path)
                H = compute_homography(pixel_points, world_points)
                overlay_frame = frames_raw[0] if frames_raw else None
                overlay_points = pixel_points
            else:
                best_frame, pixel_points, H_auto, confidence = run_static_auto_calibration(video_path)
                if confidence < 0.4:
                    print(f"⚠️ Calibración automática con baja confianza ({confidence:.2f}). Se omite salida en metros.")
                else:
                    H = H_auto
                    overlay_frame = best_frame
                    overlay_points = pixel_points
        except Exception as exc:
            print(f"⚠️ Error en calibración de cancha: {exc}")


    bounce_best_payload = None
    if args.detect_bounces and df_bounces is not None and not df_bounces.empty:
        passing_pool = df_bounces[df_bounces["passes_threshold"]] if "passes_threshold" in df_bounces.columns else df_bounces
        if not passing_pool.empty:
            best_row = passing_pool.sort_values("bounce_score", ascending=False).iloc[0]
            bounce_best_payload = {
                "frame": int(best_row["frame_bounce"]),
                "cx": float(best_row["cx"]),
                "cy": float(best_row["cy"]),
                "bounce_score": float(best_row["bounce_score"]),
            }
        else:
            best_row = df_bounces.sort_values("bounce_score", ascending=False).iloc[0]
            bounce_best_payload = {
                "frame": int(best_row["frame_bounce"]),
                "cx": float(best_row["cx"]),
                "cy": float(best_row["cy"]),
                "bounce_score": float(best_row["bounce_score"]),
            }

    render_kwargs = {}
    if args.detect_bounces and args.bounce_visuals:
        render_kwargs["bounces_df"] = df_bounces
        render_kwargs["bounce_topk"] = args.bounce_topk
        render_kwargs["bounce_best"] = bounce_best_payload
    if args.detect_hits and args.hit_visuals:
        render_kwargs["hits_df"] = df_hits
    if args.draw_players and df_players is not None and not df_players.empty:
        render_kwargs["df_players"] = df_players
    render_kwargs["draw_players"] = bool(args.draw_players)
    render_kwargs["draw_player_geometry"] = bool(args.player_geometry_visuals)
    render_kwargs["draw_ball"] = bool(args.debug_ball)
    render_kwargs["debug_players"] = bool(args.debug_players)
    if args.debug_court and court_kps is not None:
        render_kwargs["court_keypoints"] = court_kps
    if args.calibration_visuals and overlay_points is not None:
        render_kwargs["calibration_points"] = overlay_points

    render_video(
        frames_raw=frames_raw,
        df_detecciones=df_detecciones,
        df_interpolado=df_interpolado,
        output_video=output_video,
        fps=video_info.fps,
        frame_width=video_info.frame_width,
        frame_height=video_info.frame_height,
        total_frames=video_info.total_frames,
        no_gui=args.no_gui,
        **render_kwargs,
    )

    print(f"\n✅ Video guardado: {output_video}")
    print(f"📄 CSV detecciones: {output_csv}")
    print(f"📈 CSV interpolado: {output_csv_interpolado}")

    save_direction_plots(df_interpolado, output_angle_png, output_rose_png)
    print(f"📊 Gráficos generados: {output_rose_png.name}, {output_angle_png.name}")

    if H is not None:
        df_metros = apply_homography(df_interpolado, H, float(video_info.fps))
        df_metros.to_csv(output_csv_metros, index=False)
        print(f"📏 CSV en metros: {output_csv_metros}")

        if overlay_frame is not None and overlay_points is not None:
            draw_court_overlay(overlay_frame, overlay_points, output_court_overlay)
            print(f"🧭 Overlay de cancha: {output_court_overlay}")

    if args.detect_bounces:
        output_bounces.parent.mkdir(parents=True, exist_ok=True)
        output_bounce_best.parent.mkdir(parents=True, exist_ok=True)

        if df_bounces is None:
            df_bounces = detect_bounces(
                df_interpolado,
                fps=float(video_info.fps),
                smooth_window=args.bounce_smooth_window,
                min_frames_between=args.bounce_min_frames_between,
                dy_threshold_px=args.bounce_dy_threshold,
                score_threshold=args.bounce_score_threshold,
                speed_min=args.bounce_speed_min,
                min_drop_px=args.bounce_min_drop_px,
                raw_detected_frames=df_detecciones["frame"].astype(int).tolist(),
                hit_frames=(
                    df_hits["frame_hit"].astype(int).tolist()
                    if args.detect_hits and df_hits is not None and not df_hits.empty
                    else None
                ),
                exclude_post_hit=args.bounce_exclude_post_hit,
                exclude_pre_hit=args.bounce_exclude_pre_hit,
                top_k=args.bounce_topk,
            )
            if args.detect_hits and df_hits is not None and not df_hits.empty:
                df_bounces = _filter_bounces_with_next_hit(
                    df_bounces,
                    df_hits,
                    max_gap=args.bounce_max_gap_to_next_hit,
                )

        if not df_bounces.empty and H is not None:
            projected = project_points_to_meters(df_bounces[["cx", "cy"]].to_numpy(dtype="float32"), H)
            df_bounces = df_bounces.copy()
            df_bounces["X_m"] = projected[:, 0]
            df_bounces["Y_m"] = projected[:, 1]

        df_bounces.to_csv(output_bounces, index=False)
        print(f"🏀 CSV botes: {output_bounces}")

        if df_bounces.empty:
            print("⚠️ No se detectaron botes confiables.")
        else:
            passing_pool = df_bounces[df_bounces["passes_threshold"]] if "passes_threshold" in df_bounces.columns else df_bounces
            if passing_pool.empty:
                print("⚠️ No bounce candidate passed --bounce-score-threshold; keeping candidates in bounces.csv for debugging.")
            else:
                best = passing_pool.sort_values("bounce_score", ascending=False).iloc[0]
                payload = {
                    "frame": int(best["frame_bounce"]),
                    "cx": float(best["cx"]),
                    "cy": float(best["cy"]),
                    "bounce_score": float(best["bounce_score"]),
                }
                if "X_m" in df_bounces.columns and "Y_m" in df_bounces.columns:
                    payload["X_m"] = float(best["X_m"])
                    payload["Y_m"] = float(best["Y_m"])

                with output_bounce_best.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"🥇 Mejor bote: {output_bounce_best}")


if __name__ == "__main__":
    main()
