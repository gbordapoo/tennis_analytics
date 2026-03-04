from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"⚠️ Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def diagnose(outdir: Path) -> None:
    raw = _load_csv(outdir / "detecciones_ultralytics.csv")
    interp = _load_csv(outdir / "detecciones_interpoladas.csv")
    bounces = _load_csv(outdir / "bounces.csv")
    players = _load_csv(outdir / "players.csv")

    print("=== Ball trajectory diagnostics ===")
    raw_count = int(len(raw))
    last_raw = int(raw["frame"].max()) if not raw.empty and "frame" in raw.columns else None
    print(f"Raw detections: {raw_count}")
    print(f"Last raw detection frame: {last_raw}")

    if interp.empty:
        print("No interpolated trajectory found.")
    else:
        extrap_count = int(interp.get("extrapolated", False).fillna(False).astype(bool).sum())
        near_zero_speed = int(interp["speed"].fillna(0).abs().lt(1e-6).sum()) if "speed" in interp.columns else 0
        print(f"Extrapolated frames: {extrap_count}")
        print(f"Near-zero speed frames: {near_zero_speed}")

        if last_raw is not None and "frame" in interp.columns:
            tail = interp[interp["frame"] > last_raw].copy()
            if tail.empty:
                print("Trajectory does not continue after last raw frame.")
            else:
                cx_flat = tail["cx"].nunique(dropna=True) <= 1 if "cx" in tail.columns else False
                cy_flat = tail["cy"].nunique(dropna=True) <= 1 if "cy" in tail.columns else False
                print(
                    "Post-last-raw tail: "
                    f"len={len(tail)}, cx_flat={cx_flat}, cy_flat={cy_flat}, speed_mean={tail['speed'].mean():.3f}"
                )

    print("\n=== Bounce candidates ===")
    print(f"Bounces rows: {len(bounces)}")
    if not bounces.empty:
        print(f"Columns: {', '.join(bounces.columns)}")

    print("\n=== Far-player x histogram ===")
    if players.empty or "player" not in players.columns:
        print("No players.csv data available.")
        return

    far = players[players["player"] == "far"].copy()
    if far.empty:
        print("No far-player rows found.")
        return

    far["cx"] = (pd.to_numeric(far["x1"], errors="coerce") + pd.to_numeric(far["x2"], errors="coerce")) / 2.0
    gate_left = pd.to_numeric(far.get("gate_left_px"), errors="coerce") if "gate_left_px" in far.columns else None
    gate_right = pd.to_numeric(far.get("gate_right_px"), errors="coerce") if "gate_right_px" in far.columns else None

    hist = far["cx"].dropna().round(-1).value_counts().sort_index()
    print("Far cx histogram (10px bins):")
    for xbin, count in hist.items():
        print(f"  {int(xbin):>4d}px: {int(count)}")

    if gate_left is not None and gate_right is not None:
        outside = (far["cx"] < gate_left) | (far["cx"] > gate_right)
        outside_count = int(outside.fillna(False).sum())
        print(f"Far rows outside gate: {outside_count}/{len(far)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose outputs from a tennis analytics run")
    parser.add_argument("--outdir", type=str, required=True, help="Run output directory")
    args = parser.parse_args()
    diagnose(Path(args.outdir))


if __name__ == "__main__":
    main()
