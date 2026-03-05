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

        if hasattr(row, "gate_fallback"):
            val = getattr(row, "gate_fallback")
            record["gate_fallback"] = bool(val) if pd.notna(val) else False
        if hasattr(row, "gate_left_px"):
            val = getattr(row, "gate_left_px")
            record["gate_left_px"] = float(val) if pd.notna(val) else np.nan
        if hasattr(row, "gate_right_px"):
            val = getattr(row, "gate_right_px")
            record["gate_right_px"] = float(val) if pd.notna(val) else np.nan
        if hasattr(row, "foot_x"):
            val = getattr(row, "foot_x")
            record["foot_x"] = float(val) if pd.notna(val) else np.nan
        if hasattr(row, "foot_y"):
            val = getattr(row, "foot_y")
            record["foot_y"] = float(val) if pd.notna(val) else np.nan
        if hasattr(row, "track_id"):
            val = getattr(row, "track_id")
            record["track_id"] = int(val) if pd.notna(val) else -1
        if hasattr(row, "far_poly"):
            record["far_poly"] = getattr(row, "far_poly")
        if hasattr(row, "near_poly"):
            record["near_poly"] = getattr(row, "near_poly")

        players_by_frame.setdefault(int(frame), {})[str(player)] = record

    return players_by_frame




def _parse_poly(poly_text: str | float | None) -> np.ndarray | None:
    if poly_text is None or (isinstance(poly_text, float) and not np.isfinite(poly_text)):
        return None
    text = str(poly_text).strip()
    if not text or text == '[]':
        return None
    nums = np.fromstring(text.replace('[', ' ').replace(']', ' '), sep=' ')
    if nums.size < 8 or nums.size % 2 != 0:
        return None
    return nums.reshape(-1, 2).astype(np.int32)

def _draw_players(frame: np.ndarray, frame_players: dict[str, dict[str, float]], draw_player_geometry: bool = False, debug_players: bool = False) -> None:
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
        track_txt = f" id={int(pdata.get('track_id', -1))}" if debug_players and int(pdata.get('track_id', -1)) >= 0 else ""
        cv2.putText(
            frame,
            f"{player_name}{track_txt}{conf_txt}",
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

        if draw_player_geometry and np.isfinite(pdata.get("foot_x", np.nan)) and np.isfinite(pdata.get("foot_y", np.nan)):
            cv2.circle(frame, (int(round(pdata["foot_x"])), int(round(pdata["foot_y"]))), 5, color, -1)


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
    draw_player_geometry: bool = False,
    calibration_points: np.ndarray | None = None,
    bounce_best: dict[str, float | int] | None = None,
    draw_ball: bool = True,
    debug_players: bool = False,
    court_keypoints: list[float] | None = None,
) -> None:
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (frame_width, frame_height))
    print("\n🎬 Renderizando video con interpolación y extrapolación...\n")

    bounce_by_frame: dict[int, tuple[float, float, float, bool, str]] = {}
    bounce_top_candidates: list[tuple[int, float, float, float, bool, str]] = []
    best_bounce_frame: int | None = None
    if bounces_df is not None and not bounces_df.empty:
        k = max(1, int(bounce_topk))
        df_bounce_top = bounces_df.sort_values("bounce_score", ascending=False).head(k)
        for _, bounce in df_bounce_top.iterrows():
            frame_bounce = int(bounce["frame_bounce"])
            cx = float(bounce["cx"])
            cy = float(bounce["cy"])
            score = float(bounce["bounce_score"])
            btype = str(bounce.get("type", "candidate"))
            selected = bool(bounce["selected"]) if "selected" in bounces_df.columns else score >= 0.2
            bounce_top_candidates.append((frame_bounce, cx, cy, score, selected, btype))
            bounce_by_frame[frame_bounce] = (cx, cy, score, selected, btype)
        if bounce_best is not None and "frame" in bounce_best:
            best_bounce_frame = int(bounce_best["frame"])
        elif bounce_top_candidates:
            best_bounce_frame = bounce_top_candidates[0][0]

    hit_by_frame: dict[int, tuple[float, float, float]] = {}
    if hits_df is not None and not hits_df.empty:
        for _, hit in hits_df.iterrows():
            frame_hit = int(hit["frame_hit"])
            hit_by_frame[frame_hit] = (float(hit["cx"]), float(hit["cy"]), float(hit["hit_score"]))

    players_by_frame = _build_players_by_frame(df_players) if draw_players else {}
    calibration_poly: np.ndarray | None = None
    if calibration_points is not None:
        points = np.asarray(calibration_points, dtype=np.float32)
        if points.shape == (4, 2):
            calibration_poly = points.astype(np.int32)

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
        if draw_ball and not detecciones_frame.empty:
            for _, d in detecciones_frame.iterrows():
                x1, y1, x2, y2, conf = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]), float(d["confidence"])
                label = f"Ball {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                log_line += f" ✔️ Detected ({conf:.2f})"

        frame_players = players_by_frame.get(frame_id, {}) if draw_players else {}
        if draw_players and frame_players:
            _draw_players(frame, frame_players, draw_player_geometry=draw_player_geometry, debug_players=debug_players)

        if draw_players and frame_players:
            gate_left, gate_right = None, None
            far_poly = None
            near_poly = None
            gate_fallback = False
            for pdata in frame_players.values():
                if np.isfinite(pdata.get("gate_left_px", np.nan)):
                    gate_left = int(round(float(pdata["gate_left_px"])))
                if np.isfinite(pdata.get("gate_right_px", np.nan)):
                    gate_right = int(round(float(pdata["gate_right_px"])))
                gate_fallback = gate_fallback or bool(pdata.get("gate_fallback", False))
                far_poly = far_poly if far_poly is not None else _parse_poly(pdata.get("far_poly"))
                near_poly = near_poly if near_poly is not None else _parse_poly(pdata.get("near_poly"))
            if gate_left is not None and gate_right is not None:
                cv2.line(frame, (gate_left, 0), (gate_left, frame_height - 1), (0, 200, 255), 1)
                cv2.line(frame, (gate_right, 0), (gate_right, frame_height - 1), (0, 200, 255), 1)
                cv2.putText(frame, "x-gate", (gate_left + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            if gate_fallback:
                cv2.putText(frame, "WARN: gate fallback", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if draw_player_geometry:
                if far_poly is not None:
                    cv2.polylines(frame, [far_poly], True, (0, 255, 255), 2)
                if near_poly is not None:
                    cv2.polylines(frame, [near_poly], True, (255, 0, 255), 2)


        if court_keypoints is not None:
            kps = [(court_keypoints[j], court_keypoints[j + 1]) for j in range(0, min(len(court_keypoints), 28), 2)]
            for k_idx, (kx, ky) in enumerate(kps):
                if np.isfinite(kx) and np.isfinite(ky):
                    cv2.circle(frame, (int(round(kx)), int(round(ky))), 4, (50, 255, 50), -1)
                    cv2.putText(frame, str(k_idx), (int(round(kx)) + 4, int(round(ky)) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 255, 50), 1)

        if calibration_poly is not None:
            cv2.polylines(frame, [calibration_poly], True, (0, 255, 255), 2)
            for idx, (x, y) in enumerate(calibration_poly):
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(idx + 1),
                    (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

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

        for cand_idx, (bframe, bx, by, bscore, selected, btype) in enumerate(bounce_top_candidates, start=1):
            if not (np.isfinite(bx) and np.isfinite(by)):
                continue
            ix, iy = int(round(bx)), int(round(by))
            if not (0 <= ix < frame_width and 0 <= iy < frame_height):
                continue
            color = (0, 165, 255) if selected else (90, 90, 255)
            is_best = bframe == best_bounce_frame
            radius = 20 if is_best else 10
            near_frame = abs(frame_id - bframe) <= 1
            thickness = 4 if (is_best and near_frame) else (2 if near_frame else 1)
            cv2.circle(frame, (ix, iy), radius, color, thickness)
            if near_frame:
                label = f"Bounce({btype}) score={bscore:.2f}"
                cv2.putText(frame, label, (ix + 8, max(20, iy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.putText(frame, f"B{cand_idx}", (ix - 6, iy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if frame_id in bounce_by_frame:
            bx, by, bscore, _, btype = bounce_by_frame[frame_id]
            log_line += f" 🟠 Bounce {btype} ({bscore:.2f})"

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
