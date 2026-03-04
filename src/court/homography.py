from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_manual_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pixel_points = np.asarray(data["pixel_points"], dtype=np.float32)
    world_points = np.asarray(data["world_points_m"], dtype=np.float32)

    if pixel_points.shape != (4, 2):
        raise ValueError(f"pixel_points must have shape (4,2), got {pixel_points.shape}")
    if world_points.shape != (4, 2):
        raise ValueError(f"world_points_m must have shape (4,2), got {world_points.shape}")

    return pixel_points, world_points


def compute_homography(pixel_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    pixel_points = np.asarray(pixel_points, dtype=np.float32)
    world_points = np.asarray(world_points, dtype=np.float32)

    if pixel_points.shape != (4, 2) or world_points.shape != (4, 2):
        raise ValueError("compute_homography expects pixel_points and world_points with shape (4,2)")

    H = cv2.getPerspectiveTransform(pixel_points, world_points)
    return H


def apply_homography(df_interpolado: pd.DataFrame, H: np.ndarray, fps: float) -> pd.DataFrame:
    if fps <= 0:
        raise ValueError("fps must be > 0")

    required_cols = {"frame", "cx", "cy"}
    missing = required_cols - set(df_interpolado.columns)
    if missing:
        raise ValueError(f"Missing required columns in df_interpolado: {sorted(missing)}")

    df_out = df_interpolado.copy()

    points_px = df_out[["cx", "cy"]].to_numpy(dtype=np.float32).reshape(-1, 1, 2)
    points_m = cv2.perspectiveTransform(points_px, H).reshape(-1, 2)

    df_out["X_m"] = points_m[:, 0]
    df_out["Y_m"] = points_m[:, 1]
    df_out["dX_m"] = df_out["X_m"].diff().fillna(0.0)
    df_out["dY_m"] = df_out["Y_m"].diff().fillna(0.0)

    dt = 1.0 / fps
    df_out["speed_mps"] = np.sqrt(df_out["dX_m"] ** 2 + df_out["dY_m"] ** 2) / dt
    df_out["speed_kmh"] = df_out["speed_mps"] * 3.6

    return df_out


def project_points_to_meters(pixel_points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    pixel_points: shape (N,2) float32 (cx, cy)
    returns: shape (N,2) float32 (X_m, Y_m)
    usa cv2.perspectiveTransform
    """
    pts = np.asarray(pixel_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pixel_points must have shape (N,2), got {pts.shape}")

    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
