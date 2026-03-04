# ARCHITECTURE

## 1) System diagram

```text
src/main.py (CLI + orchestration)
  |
  +--> ball.detect.load_model
  +--> ball.detect.run_detection
  |       -> detecciones_ultralytics.csv
  |
  +--> ball.track.interpolar_detecciones
  |       -> detecciones_interpoladas.csv
  |
  +--> [optional] player.pose.ensure_pose_model + detect_players
  |       -> players.csv
  |
  +--> [optional] ball.hit.detect_hits
  |       -> hits.csv (or --hit-out)
  |
  +--> [optional] ball.bounce.detect_bounces
  |       -> bounces.csv (or --bounce-out)
  |       -> bounce_best.json (or --bounce-best-out)
  |
  +--> [optional] court calibration
  |       |- auto/static: court.auto_calibrate.run_static_auto_calibration
  |       `- manual: court.homography.load_manual_calibration + compute_homography
  |          -> H (homography)
  |
  +--> viz.render.render_video
  |       -> output_ultralytics.mp4
  |
  +--> viz.render.save_direction_plots
  |       -> grafico_angulo_por_frame.png, rosa_direcciones.png
  |
  `--> [if H] court.homography.apply_homography
          -> detecciones_metros.csv
          -> (optional) court.auto_calibrate.draw_court_overlay -> court_overlay.png
```

## 2) Execution flow (entrypoint to outputs)

Primary entrypoint is `src/main.py` (`main()`).

1. Parse CLI args (`parse_args`).
2. Resolve paths (`_resolve_path`) against repo root / script dir.
3. Validate required files (`_ensure_file`) for model and video.
4. Ball inference:
   - `load_model(model_path)`
   - `run_detection(model, video_path)` -> `frames_raw`, `df_detecciones`, `VideoInfo`.
   - Save `df_detecciones` to CSV.
5. Build continuous ball trajectory:
   - `interpolar_detecciones(df_detecciones, total_frames, extrap_frames)`.
   - Save interpolated CSV.
6. Optional player/hit stages:
   - if `--detect-players` or `--detect-hits`: `ensure_pose_model` then `detect_players` and save `players.csv`.
   - if `--detect-hits`: `detect_hits(df_interpolado, df_players)` and save hits CSV.
7. Optional bounce stage:
   - `detect_bounces(...)` using interpolated trajectory (+ optional hit frame exclusion).
   - optional `_filter_bounces_with_next_hit(...)` in `main.py`.
8. Optional calibration stage:
   - manual mode: `load_manual_calibration` + `compute_homography`.
   - otherwise auto/static: `run_static_auto_calibration`.
9. Render video overlays with `viz.render.render_video(...)`.
10. Save angle/polar plots with `save_direction_plots(...)`.
11. If homography `H` exists:
   - `apply_homography(...)` for metric trajectory CSV.
   - optionally `draw_court_overlay(...)`.
12. If bounce detection enabled:
   - save bounce CSV and best-bounce JSON payload.
   - when `H` exists, add `X_m`, `Y_m` for bounce points using `project_points_to_meters`.

## 3) Data contracts

### `VideoInfo` dataclass (`src/ball/detect.py`)

- `fps: float`
- `frame_width: int`
- `frame_height: int`
- `total_frames: int`

### `df_detecciones` (raw detections)

Written by `run_detection` with columns:
- `frame` (1-based)
- `x1`, `y1`, `x2`, `y2`
- `confidence`

### `df_interpolado` (dense ball trajectory)

Written by `interpolar_detecciones` with columns:
- `frame`, `cx`, `cy`
- `vx`, `vy`, `speed`, `angle`
- `extrapolated` (bool)

### `df_players`

Written by `detect_players` with columns:
- `frame`, `player` (`near`/`far`)
- `x1`, `y1`, `x2`, `y2`
- `wrist_l_x`, `wrist_l_y`, `wrist_r_x`, `wrist_r_y`
- `conf`

### `df_hits`

Written by `detect_hits` with columns:
- `frame_hit`, `player`, `cx`, `cy`, `hit_score`

### `df_bounces`

Written by `detect_bounces` with columns:
- `frame_bounce`, `cx`, `cy`, `bounce_score`
- optional enrichment in `main.py`: `X_m`, `Y_m` when homography is available.

### Manual calibration JSON

Loaded by `load_manual_calibration(path)`:
- required key `pixel_points`: `(4,2)`
- required key `world_points_m`: `(4,2)`
- `court_type` may exist (from click tool) but is not required by loader.

## 4) Module-by-module breakdown

### `src/main.py`

- Responsibilities:
  - CLI definition, path resolution, orchestration, output persistence.
- Key functions:
  - `parse_args()`: defines all main CLI flags.
  - `_resolve_path()`: robust relative path handling.
  - `_build_player_selection_cfg()`: x-gate config + optional calibration gate values.
  - `_filter_bounces_with_next_hit()`: temporal filtering of bounce candidates.
  - `main()`: full pipeline.
- Inputs/outputs:
  - Inputs: CLI, model/video files, optional calibration/pose files.
  - Outputs: MP4, CSV, JSON, PNG artifacts.
- Coupling:
  - imports every domain package (`ball`, `court`, `player`, `viz`).

### `src/ball/detect.py`

- Responsibilities: model load + per-frame ball inference.
- Key symbols:
  - `VideoInfo`, `load_model`, `run_detection`.
- Dependencies: `ultralytics.YOLO`, `cv2`, `pandas`.

### `src/ball/track.py`

- Responsibilities: interpolate sparse detections into full trajectory.
- Key symbol: `interpolar_detecciones`.
- Dependencies: `numpy`, `pandas`.

### `src/ball/bounce.py`

- Responsibilities: bounce candidate generation/scoring/temporal suppression.
- Key symbol: `detect_bounces`.
- Dependencies: `numpy`, `pandas`.

### `src/ball/hit.py`

- Responsibilities: detect hits using angle change and player-distance logic.
- Key symbol: `detect_hits`.
- Dependencies: `numpy`, `pandas`, `math`; expects player DataFrame schema from `player.pose`.

### `src/player/pose.py`

- Responsibilities: detect people with pose model; assign near/far robustly.
- Key symbols:
  - `ensure_pose_model`: fail-fast if local pose weights missing.
  - `detect_players`: model inference + output rows.
  - `select_near_far_people`: gating + temporal consistency.
- Dependencies: `ultralytics.YOLO` (lazy import), `numpy`, `pandas`.

### `src/court/auto_calibrate.py`

- Responsibilities: auto court-corner estimation and confidence scoring.
- Key symbols:
  - `pick_best_frame`, `detect_court_corners`, `run_static_auto_calibration`, `draw_court_overlay`.
- Dependencies: `cv2`, `numpy`, `court.homography.compute_homography`.

### `src/court/homography.py`

- Responsibilities: manual calibration load + perspective transforms.
- Key symbols:
  - `load_manual_calibration`, `compute_homography`, `apply_homography`, `project_points_to_meters`.
- Dependencies: `json`, `cv2`, `numpy`, `pandas`.

### `src/court/calibrate_click.py`

- Responsibilities: interactive click-based calibration file generation.
- Key symbols: `parse_args`, `collect_four_points`, `prompt_court_type`, `main`.
- Dependencies: `argparse`, `cv2`, `json`.

### `src/viz/render.py`

- Responsibilities: overlay rendering and static plots.
- Key symbols:
  - `render_video`: frame overlays for ball/players/hits/bounces/calibration.
  - `save_direction_plots`: line + polar plots from `angle`.
- Dependencies: `cv2`, `matplotlib`, `numpy`, `pandas`.

### `scripts/debug_frame.py`

- Responsibilities: save one debug frame with overlays + optional sideline assertion.
- Dependencies: `argparse`, `cv2`, `numpy`, `pandas`.

## 5) Full CLI reference

### `python src/main.py`

**Core**
- `--model` (default: `../models/best.pt`) path to ball YOLO weights.
- `--video` (default: `../videos/federer_murray_trim.mp4`) input video.
- `--outdir` (default: `../outputs`) output directory.
- `--extrap` (default: `5`) backward extrapolation frames.
- `--no-gui` (flag) disable `cv2.imshow`.

**Calibration**
- `--auto-calibrate` (flag) enable calibration flow.
- `--calibration-mode` (default: `auto`, choices: `auto|static|manual`) calibration strategy.
- `--calibration` (default: `None`) manual calibration JSON path.
- `--calibration-visuals` (flag) draw calibration polygon/points in rendered video.

**Bounce detection**
- `--detect-bounces` (flag) run bounce detector.
- `--bounce-visuals` (flag) draw bounce markers.
- `--bounce-topk` (default: `3`) max bounce candidates to draw.
- `--bounce-smooth-window` (default: `5`) smoothing window size.
- `--bounce-min-frames-between` (default: `8`) temporal suppression gap.
- `--bounce-dy-threshold` / `--bounce-dy-threshold-px` (default: `1.0`) sign-flip threshold.
- `--bounce-score-threshold` (default: `0.2`) minimum candidate score.
- `--bounce-speed-min` (default: `2.0`) minimum speed for reliable bounce frames.
- `--bounce-min-drop-px` (default: `6.0`) minimum drop/rise around local minimum.
- `--bounce-exclude-post-hit` (default: `4`) exclude frames after hit.
- `--bounce-exclude-pre-hit` (default: `0`) exclude frames before hit.
- `--bounce-out` (default: `bounces.csv`) bounce CSV output path.
- `--bounce-best-out` (default: `bounce_best.json`) best bounce JSON path.
- `--bounce-max-gap-to-next-hit` (default: `15`) post-filter max gap from bounce to next hit.

**Players**
- `--detect-players` (flag) run player pose detection.
- `--draw-players` (default: `True`) draw near/far boxes.
- `--no-draw-players` (flag) disable player drawing.
- `--pose-model` (default: `models/yolov8n-pose.pt`) pose model path.
- `--player-xgate-left` (default: `0.18`) left x-fraction gate.
- `--player-xgate-right` (default: `0.82`) right x-fraction gate.
- `--player-calibration-margin-px` (default: `20.0`) calibration-gate margin.
- `--player-min-conf` (default: `0.3`) minimum person confidence for near/far assignment.
- `--pose-download` (flag, suppressed help) currently parsed but not used.

**Hit detection**
- `--detect-hits` (flag) run hit detector.
- `--hit-out` (default: `hits.csv`) hit CSV output path.
- `--hit-visuals` (flag) draw hit markers.

### `python src/court/calibrate_click.py`
- `--video` (required): input video.
- `--output` (default: `calibration.json`): calibration JSON path.

### `python scripts/debug_frame.py`
- `--video` (required): input video.
- `--frame` (required): 1-based frame to render.
- `--outdir` (required): output root (writes in `<outdir>/debug/`).
- `--players-csv` (default: `None`): player CSV path.
- `--ball-csv` (default: `None`): ball trajectory CSV path.
- `--assert-far-not-sideline` (flag): assert far player foot_x stays inside x-gate.
- `--xgate-left` (default: `0.18`) assertion gate left fraction.
- `--xgate-right` (default: `0.82`) assertion gate right fraction.

## 6) Configuration points

- CLI-to-algorithm mappings:
  - bounce thresholds/top-k in `src/main.py` map directly into `detect_bounces(...)`.
  - player x-gates in `src/main.py` map into `select_near_far_people(...)` via config dict.
  - extrapolation window from `--extrap` maps into `interpolar_detecciones(..., extrap_frames=...)`.
- Non-CLI algorithm defaults:
  - `detect_hits(...)` thresholds (`dist_px`, angle threshold, score threshold, etc.) are function defaults in `src/ball/hit.py`.
  - auto-calibration acceptance threshold is in `src/main.py` (`confidence < 0.4` => skip metric output).

Safe changes:
- Prefer exposing changed thresholds through CLI instead of silently changing function defaults.
- Keep DataFrame column names stable to preserve render and output compatibility.

## 7) Extension points

- Add new event detector:
  - create module under `src/ball/` (or new package), return explicit DataFrame schema, call it from `src/main.py`, then add serialization + optional overlay.
- Add new visual layer:
  - extend `render_video(...)` in `src/viz/render.py` with optional DataFrame and frame-indexed lookup.
- Add new calibration strategy:
  - add a new branch in `src/main.py` calibration section and implement helper under `src/court/`.

## 8) Performance notes

- Hotspots:
  - YOLO inference over all frames for ball (`run_detection`).
  - Separate YOLO inference pass for players (`detect_players`) when enabled.
  - Full-frame rendering loop (`render_video`) + optional GUI wait.
- Current optimizations present:
  - Reuse `df_players` if already computed for hits.
  - Skip optional expensive branches unless enabled by flags.
- Constraints:
  - `frames_raw` stores all frames in memory (simple architecture, higher memory footprint on long videos).
