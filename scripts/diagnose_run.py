from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick diagnostics for a pipeline output directory")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory from src/main.py")
    parser.add_argument("--bounce-speed-min", type=float, default=2.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    raw_path = outdir / "detecciones_ultralytics.csv"
    interp_path = outdir / "detecciones_interpoladas.csv"

    if not raw_path.exists() or not interp_path.exists():
        raise FileNotFoundError("Missing detecciones_ultralytics.csv or detecciones_interpoladas.csv in outdir")

    raw = pd.read_csv(raw_path)
    interp = pd.read_csv(interp_path)

    raw_count = len(raw)
    raw_unique = raw["frame"].nunique() if "frame" in raw.columns else 0
    print(f"raw detections: {raw_count} (unique frames: {raw_unique})")

    interp_dups = int(interp.duplicated(subset=["frame"]).sum()) if "frame" in interp.columns else -1
    print(f"interpolated duplicate frames: {interp_dups}")

    if "frame" in raw.columns and not raw.empty:
        print(f"last frame with raw detection: {int(raw['frame'].max())}")

    extrap_count = int(interp["extrapolated"].fillna(False).astype(bool).sum()) if "extrapolated" in interp.columns else 0
    print(f"extrapolated frames: {extrap_count}")

    if {"speed", "extrapolated"}.issubset(interp.columns):
        speed = pd.to_numeric(interp["speed"], errors="coerce")
        extrap = interp["extrapolated"].fillna(False).astype(bool)
        filtered = extrap | (speed < float(args.bounce_speed_min))
        print(f"speed stats: min={speed.min():.3f}, median={speed.median():.3f}, max={speed.max():.3f}")
        print(f"frames filtered by extrapolated/speed<{args.bounce_speed_min}: {int(filtered.sum())}")


if __name__ == "__main__":
    main()
