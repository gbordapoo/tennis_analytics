# PROJECT_SUMMARY

## What this project does

This repository processes tennis match video using YOLO-based inference and rule-based post-processing. The main script (`src/main.py`) detects the ball per frame, interpolates/extrapolates a continuous trajectory, optionally detects players (near/far) with pose keypoints, optionally detects hits and bounces, optionally estimates court homography (auto or manual), and writes annotated media + tabular outputs.

## Capabilities (implemented)

- Ball detection from each frame (`src/ball/detect.py`).
- Trajectory interpolation + backward extrapolation (`src/ball/track.py`).
- Bounce detection from vertical motion sign changes and scoring (`src/ball/bounce.py`).
- Hit detection from trajectory angle change + player proximity (`src/ball/hit.py`).
- Player detection with YOLO pose + near/far assignment + wrists (`src/player/pose.py`).
- Court calibration:
  - static auto-calibration from line heuristics (`src/court/auto_calibrate.py`)
  - manual calibration JSON loader + click tool (`src/court/homography.py`, `src/court/calibrate_click.py`)
- Pixel→meter projection and speed estimation (`src/court/homography.py`).
- Visualization outputs (annotated video, direction plots, court overlay) (`src/viz/render.py`).

## Typical run commands

Run from repo root.

- Minimal run:
  - `python src/main.py --model models/yolo5_last.pt --video videos/federer_murray_trim.mp4 --outdir outputs`
- Headless/no GUI:
  - `python src/main.py --model models/yolo5_last.pt --video videos/federer_murray_trim.mp4 --outdir outputs --no-gui`
- Players + hits:
  - `python src/main.py --model models/yolo5_last.pt --video videos/federer_murray_trim.mp4 --outdir outputs --no-gui --detect-players --detect-hits --hit-visuals`
- Full features (players/hits/bounces/calibration):
  - `python src/main.py --model models/yolo5_last.pt --video videos/federer_murray_trim.mp4 --outdir outputs --no-gui --detect-players --detect-hits --hit-visuals --detect-bounces --bounce-visuals --auto-calibrate --calibration-visuals`
- Manual calibration helper:
  - `python src/court/calibrate_click.py --video videos/federer_murray_trim.mp4 --output outputs/calibration.json`

## Inputs and outputs

### Inputs

- Ball model weights (`--model`, default `models/yolo5_last.pt`).
- Input video (`--video`, default `../videos/federer_murray_trim.mp4`).
- Optional pose model (`--pose-model`, default `models/yolov8n-pose.pt`) for players/hits.
- Optional manual calibration JSON (`--calibration`) for `--calibration-mode manual`.

Video extension is not hardcoded; OpenCV `VideoCapture` is used. Existing commands and outputs assume MP4.

### Outputs (under `--outdir` unless absolute output paths are passed)

- `output_ultralytics.mp4`
- `detecciones_ultralytics.csv`
- `detecciones_interpoladas.csv`
- `detecciones_metros.csv` (only if homography available)
- `players.csv` (if players/hits enabled)
- `hits.csv` (or `--hit-out`, if hits enabled)
- `bounces.csv` + `bounce_best.json` (or custom bounce outputs, if bounces enabled)
- `grafico_angulo_por_frame.png`
- `rosa_direcciones.png`
- `court_overlay.png` (if calibration points available)

## Repo layout (top-level)

- `src/` — pipeline code.
  - `src/main.py` orchestrator and primary CLI.
  - `src/ball/` ball detection/track/hit/bounce logic.
  - `src/player/` pose and near/far selection.
  - `src/court/` calibration + homography helpers.
  - `src/viz/` video rendering + plots.
- `scripts/` — debug helpers (`scripts/debug_frame.py`).
- `videos/` — input video placeholder (`.gitkeep`).
- `requirements.txt` — runtime Python dependencies.

## Quick troubleshooting

- **`Model file not found`**: verify `--model` path; prefer absolute path if uncertain.
- **`Video file not found` or `Could not open video`**: verify `--video`, codec support, and OpenCV backend.
- **Pose model missing** (`Pose model not found at models/yolov8n-pose.pt`): download manually; code does not auto-download pose weights.
- **No detections found**: validate ball model quality/domain match for the specific video.
- **No bounces/hits detected**: adjust bounce CLI thresholds (`--bounce-speed-min`, `--bounce-min-drop-px`, `--bounce-score-threshold`); hit thresholds are currently hardcoded in `detect_hits(...)`.
- **No metric output (`detecciones_metros.csv`)**: calibration may fail or confidence may be `< 0.4` in auto mode.
- **GUI errors on server/headless machine**: add `--no-gui`.
- **Imports fail (`No module named ball/...`)**: run from repo root as `python src/main.py`.
- **Unexpected output path for hits/bounces**: relative `--hit-out`, `--bounce-out`, `--bounce-best-out` are joined under `--outdir`.
- **Manual calibration rejected**: ensure JSON has `pixel_points` and `world_points_m`, each shape `(4,2)`.


- Court keypoints model path: `models/keypoints_model.pth` (loaded once on first frame and reused).
