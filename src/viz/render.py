from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _build_players_by_frame(df_players: pd.DataFrame | None) -> dict[int, dict[str, dict[str, float]]]:
    if df_players is None or df_players.empty:
        return {}

    required = {"frame", "player", "x1", "y1", "x2", "y2"}
    if not required.issubset(df_players.columns):
        return {}

    players_by_frame: dict[int, dict[str, dict[str, float]]] = {}
    wrist_cols = ["wrist_l_x", "wrist_l_y", "wrist_r_x", "wrist_r_y"]

    for row in df_players.itertuples(index=False):
        frame = getattr(row, "frame", None)
        player = getattr(row, "player", None)
        if pd.isna(frame) or player not in {"near", "far"}:
            continue

        record = {
            "x1": float(getattr(row, "x1")),
            "y1": float(getattr(row, "y1")),
            "x2": float(getattr(row, "x2")),
            "y2": float(getattr(row, "y2")),
            "conf": float(getattr(row, "conf")) if hasattr(row, "conf") and pd.notna(getattr(row, "conf")) else np.nan,
        }
        for col in wrist_cols:
            val = getattr(row, col) if hasattr(row, col) else np.nan
            record[col] = float(val) if pd.notna(val) else np.nan

        players_by_frame.setdefault(int(frame), {})[str(player)] = record

    return players_by_frame


def _draw_players(frame: np.ndarray, frame_players: dict[str, dict[str, float]]) -> None:
    colors = {
        "near": (255, 0, 255),
        "far": (0, 255, 255),
    }

    for player_name in ("near", "far"):
        pdata = frame_players.get(player_name)
        if not pdata:
            continue

        x1, y1, x2, y2 = (int(round(pdata["x1"])), int(round(pdata["y1"])), int(round(pdata["x2"])), int(round(pdata["y2"])))
        color = colors[player_name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        conf_txt = f" {pdata['conf']:.2f}" if np.isfinite(pdata.get("conf", np.nan)) else ""
        cv2.putText(
            frame,
            f"{player_name}{conf_txt}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        wrists = [
            (pdata.get("wrist_l_x", np.nan), pdata.get("wrist_l_y", np.nan)),
            (pdata.get("wrist_r_x", np.nan), pdata.get("wrist_r_y", np.nan)),
        ]
        for wx, wy in wrists:
            if np.isfinite(wx) and np.isfinite(wy):
                cv2.circle(frame, (int(round(wx)), int(round(wy))), 4, color, -1)


def render_video(
    frames_raw: list,
    df_detecciones: pd.DataFrame,
    df_interpolado: pd.DataFrame,
    output_video: Path,
    fps: float,
    frame_width: int,
    frame_height: int,
    total_frames: int,
    no_gui: bool,
    bounces_df: pd.DataFrame | None = None,
    hits_df: pd.DataFrame | None = None,
    bounce_topk: int = 3,
    df_players: pd.DataFrame | None = None,
    draw_players: bool = True,
) -> None:
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (frame_width, frame_height))
    print("\n🎬 Renderizando video con interpolación y extrapolación...\n")

    bounce_by_frame: dict[int, tuple[float, float, float]] = {}
    if bounces_df is not None and not bounces_df.empty:
        k = max(1, int(bounce_topk))
        df_bounce_top = bounces_df.sort_values("bounce_score", ascending=False).head(k)
        for _, bounce in df_bounce_top.iterrows():
            frame_bounce = int(bounce["frame_bounce"])
            cx = float(bounce["cx"])
            cy = float(bounce["cy"])
            score = float(bounce["bounce_score"])
            bounce_by_frame[frame_bounce] = (cx, cy, score)

    hit_by_frame: dict[int, tuple[float, float, float]] = {}
    if hits_df is not None and not hits_df.empty:
        for _, hit in hits_df.iterrows():
            frame_hit = int(hit["frame_hit"])
            hit_by_frame[frame_hit] = (float(hit["cx"]), float(hit["cy"]), float(hit["hit_score"]))

    players_by_frame = _build_players_by_frame(df_players) if draw_players else {}

    for i, frame in enumerate(frames_raw):
        frame_id = i + 1
        log_line = f"[Frame {frame_id}/{total_frames}] "

        cv2.putText(frame, f"Frame: {frame_id}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        row = df_interpolado[df_interpolado["frame"] == frame_id]
        if not row.empty:
            cx, cy = int(row["cx"].values[0]), int(row["cy"].values[0])
            vx, vy = float(row["vx"].values[0]), float(row["vy"].values[0])
            fx, fy = int(cx + vx * 2), int(cy + vy * 2)
            extrap = bool(row["extrapolated"].values[0])

            if extrap:
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                log_line += "• Extrapolado"
            else:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                log_line += "• Interpolado"
            cv2.arrowedLine(frame, (cx, cy), (fx, fy), (255, 255, 255), 2, tipLength=0.3)

        detecciones_frame = df_detecciones[df_detecciones["frame"] == frame_id]
        if not detecciones_frame.empty:
            for _, d in detecciones_frame.iterrows():
                x1, y1, x2, y2, conf = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]), float(d["confidence"])
                label = f"Ball {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                log_line += f" ✔️ Detected ({conf:.2f})"

        if draw_players and frame_id in players_by_frame:
            _draw_players(frame, players_by_frame[frame_id])

        if frame_id in hit_by_frame:
            hx, hy, hscore = hit_by_frame[frame_id]
            if np.isfinite(hx) and np.isfinite(hy):
                ix, iy = int(round(hx)), int(round(hy))
                if 0 <= ix < frame_width and 0 <= iy < frame_height:
                    cv2.circle(frame, (ix, iy), 18, (255, 0, 0), 3)
                    cv2.putText(
                        frame,
                        f"HIT {hscore:.2f}",
                        (ix + 10, max(30, iy - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
                    log_line += f" 🔵 Hit ({hscore:.2f})"

        if frame_id in bounce_by_frame:
            bx, by, bscore = bounce_by_frame[frame_id]
            if np.isfinite(bx) and np.isfinite(by):
                ix, iy = int(round(bx)), int(round(by))
                if 0 <= ix < frame_width and 0 <= iy < frame_height:
                    cv2.circle(frame, (ix, iy), 25, (0, 165, 255), 4)
                    cv2.putText(
                        frame,
                        f"BOUNCE {bscore:.2f}",
                        (ix + 12, max(30, iy - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )
                    log_line += f" 🟠 Bounce ({bscore:.2f})"

        print(log_line)
        out.write(frame)

        if not no_gui:
            cv2.imshow("Ball Detection + Interpolation", frame)
            key = cv2.waitKey(int(1000 / float(fps)))
            if key == 27:
                break

    out.release()
    if not no_gui:
        cv2.destroyAllWindows()


def save_direction_plots(df_interpolado: pd.DataFrame, output_angle_png: Path, output_rose_png: Path) -> None:
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
