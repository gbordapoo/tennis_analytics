from __future__ import annotations

import argparse
from pathlib import Path

from ball.detect import load_model, run_detection
from ball.track import interpolar_detecciones
from court.auto_calibrate import draw_court_overlay, run_static_auto_calibration
from court.homography import apply_homography, compute_homography, load_manual_calibration
from viz.render import render_video, save_direction_plots


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Tennis Ball Detection + Interpolation/Extrapolation")
    parser.add_argument("--model", type=str, default="../models/best.pt", help="Path to YOLOv8 .pt model")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Make paths robust regardless of where you run from
    script_dir = Path(__file__).resolve().parent
    model_path = (script_dir / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model).resolve()
    video_path = (script_dir / args.video).resolve() if not Path(args.video).is_absolute() else Path(args.video).resolve()
    outdir = (script_dir / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir).resolve()

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

    model = load_model(model_path)
    frames_raw, df_detecciones, video_info = run_detection(model, video_path)
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
    )

    print(f"\n✅ Video guardado: {output_video}")
    print(f"📄 CSV detecciones: {output_csv}")
    print(f"📈 CSV interpolado: {output_csv_interpolado}")

    save_direction_plots(df_interpolado, output_angle_png, output_rose_png)
    print(f"📊 Gráficos generados: {output_rose_png.name}, {output_angle_png.name}")

    if args.auto_calibrate:
        try:
            H = None
            overlay_frame = None
            overlay_points = None

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

            if H is not None:
                df_metros = apply_homography(df_interpolado, H, float(video_info.fps))
                df_metros.to_csv(output_csv_metros, index=False)
                print(f"📏 CSV en metros: {output_csv_metros}")

                if overlay_frame is not None and overlay_points is not None:
                    draw_court_overlay(overlay_frame, overlay_points, output_court_overlay)
                    print(f"🧭 Overlay de cancha: {output_court_overlay}")
        except Exception as exc:
            print(f"⚠️ Error en calibración de cancha: {exc}")


if __name__ == "__main__":
    main()
