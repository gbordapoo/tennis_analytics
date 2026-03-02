from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def interpolar_detecciones(df_detecciones: pd.DataFrame, total_frames: int | None = None, extrap_frames: int = 5) -> pd.DataFrame:
    df_detecciones = df_detecciones.copy()

    df_detecciones["cx"] = (df_detecciones["x1"] + df_detecciones["x2"]) / 2
    df_detecciones["cy"] = (df_detecciones["y1"] + df_detecciones["y2"]) / 2

    start = int(df_detecciones["frame"].min())
    end = int(total_frames) if total_frames is not None else int(df_detecciones["frame"].max())
    all_frames = pd.DataFrame({"frame": range(start, end + 1)})

    df_full = pd.merge(all_frames, df_detecciones[["frame", "cx", "cy"]], on="frame", how="left")
    df_full["cx"] = df_full["cx"].interpolate().bfill()
    df_full["cy"] = df_full["cy"].interpolate().bfill()

    df_full["vx"] = df_full["cx"].diff().fillna(0)
    df_full["vy"] = df_full["cy"].diff().fillna(0)
    df_full["speed"] = np.sqrt(df_full["vx"] ** 2 + df_full["vy"] ** 2)
    df_full["angle"] = np.degrees(np.arctan2(df_full["vy"], df_full["vx"]))

    # EXTRAPOLACIÓN HACIA ATRÁS
    extrap_rows = []
    first_row = df_full.iloc[0]
    cx, cy = float(first_row["cx"]), float(first_row["cy"])
    vx, vy = float(first_row["vx"]), float(first_row["vy"])

    for i in range(1, extrap_frames + 1):
        frame_id = int(first_row["frame"]) - i
        if frame_id < 1:
            break
        new_cx = cx - vx * i
        new_cy = cy - vy * i
        extrap_rows.append(
            {
                "frame": frame_id,
                "cx": new_cx,
                "cy": new_cy,
                "vx": vx,
                "vy": vy,
                "speed": float(np.sqrt(vx**2 + vy**2)),
                "angle": float(np.degrees(np.arctan2(vy, vx))),
                "extrapolated": True,
            }
        )

    df_full["extrapolated"] = False
    df_extrap = pd.DataFrame(extrap_rows)
    df_resultado = pd.concat([df_extrap, df_full], ignore_index=True).sort_values(by="frame")
    df_resultado.reset_index(drop=True, inplace=True)

    return df_resultado


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
    output_angle_png = outdir / "grafico_angulo_por_frame.png"
    output_rose_png = outdir / "rosa_direcciones.png"

    # Cargar modelo
    model = YOLO(str(model_path))

    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # fallback

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_raw = []
    detecciones = []

    # Primera pasada: detección
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frames_raw.append(frame.copy())

        results = model(frame, verbose=False)
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    detecciones.append(
                        {
                            "frame": frame_count,
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "confidence": round(conf, 4),
                        }
                    )

    cap.release()

    df_detecciones = pd.DataFrame(detecciones)
    df_detecciones.to_csv(output_csv, index=False)

    if df_detecciones.empty:
        print("⚠️ No se encontraron detecciones.")
        return

    # Interpolación + extrapolación
    df_interpolado = interpolar_detecciones(df_detecciones, total_frames=total_frames, extrap_frames=int(args.extrap))
    df_interpolado.to_csv(output_csv_interpolado, index=False)

    # Segunda pasada: render
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (frame_width, frame_height))
    print("\n🎬 Renderizando video con interpolación y extrapolación...\n")

    for i, frame in enumerate(frames_raw):
        frame_id = i + 1
        log_line = f"[Frame {frame_id}/{total_frames}] "

        # Mostrar número de frame
        cv2.putText(frame, f"Frame: {frame_id}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        row = df_interpolado[df_interpolado["frame"] == frame_id]
        if not row.empty:
            cx, cy = int(row["cx"].values[0]), int(row["cy"].values[0])
            vx, vy = float(row["vx"].values[0]), float(row["vy"].values[0])
            fx, fy = int(cx + vx * 2), int(cy + vy * 2)
            extrap = bool(row["extrapolated"].values[0])

            # Punto extrapolado en rojo
            if extrap:
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                log_line += "• Extrapolado"
            else:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                log_line += "• Interpolado"
            cv2.arrowedLine(frame, (cx, cy), (fx, fy), (255, 255, 255), 2, tipLength=0.3)

        # Detecciones reales
        detecciones_frame = df_detecciones[df_detecciones["frame"] == frame_id]
        if not detecciones_frame.empty:
            for _, d in detecciones_frame.iterrows():
                x1, y1, x2, y2, conf = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]), float(d["confidence"])
                label = f"Ball {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                log_line += f" ✔️ Detected ({conf:.2f})"

        print(log_line)
        out.write(frame)

        if not args.no_gui:
            cv2.imshow("Ball Detection + Interpolation", frame)
            key = cv2.waitKey(int(1000 / float(fps)))
            if key == 27:  # ESC
                break

    out.release()
    if not args.no_gui:
        cv2.destroyAllWindows()

    print(f"\n✅ Video guardado: {output_video}")
    print(f"📄 CSV detecciones: {output_csv}")
    print(f"📈 CSV interpolado: {output_csv_interpolado}")

    # Gráficos de análisis
    plt.figure(figsize=(10, 4))
    plt.plot(df_interpolado["frame"], df_interpolado["angle"], marker="o", linestyle="-", color="tab:blue")
    plt.title("Ángulo de la pelota por frame")
    plt.xlabel("Frame")
    plt.ylabel("Ángulo (grados)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_angle_png)
    plt.close()

    angles_rad = np.radians(df_interpolado["angle"])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    bins = 36
    counts, bin_edges = np.histogram(angles_rad, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centers, counts, width=2 * np.pi / bins, bottom=0.0, color="skyblue", edgecolor="black")
    ax.set_title("Distribución de ángulos de movimiento", va="bottom")
    plt.tight_layout()
    plt.savefig(output_rose_png)
    plt.close()

    print(f"📊 Gráficos generados: {output_rose_png.name}, {output_angle_png.name}")


if __name__ == "__main__":
    main()