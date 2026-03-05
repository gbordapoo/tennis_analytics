from __future__ import annotations

import numpy as np
import pandas as pd


def interpolar_detecciones(df_detecciones: pd.DataFrame, total_frames: int | None = None, extrap_frames: int = 5) -> pd.DataFrame:
    df_detecciones = df_detecciones.copy()

    # Ensure exactly one detection per frame before interpolation.
    if "confidence" in df_detecciones.columns:
        df_detecciones = (
            df_detecciones.sort_values("confidence", ascending=False)
            .drop_duplicates(subset=["frame"], keep="first")
            .sort_values("frame")
            .reset_index(drop=True)
        )
    else:
        df_detecciones = df_detecciones.drop_duplicates(subset=["frame"], keep="first").sort_values("frame").reset_index(drop=True)

    df_detecciones["cx"] = (df_detecciones["x1"] + df_detecciones["x2"]) / 2
    df_detecciones["cy"] = (df_detecciones["y1"] + df_detecciones["y2"]) / 2

    start = int(df_detecciones["frame"].min())
    end = int(total_frames) if total_frames is not None else int(df_detecciones["frame"].max())
    all_frames = pd.DataFrame({"frame": range(start, end + 1)})

    unique_track = df_detecciones[["frame", "cx", "cy"]].drop_duplicates(subset=["frame"], keep="first")
    df_full = pd.merge(all_frames, unique_track, on="frame", how="left")
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
    df_resultado = df_resultado.drop_duplicates(subset=["frame"], keep="first")
    df_resultado.reset_index(drop=True, inplace=True)

    return df_resultado
