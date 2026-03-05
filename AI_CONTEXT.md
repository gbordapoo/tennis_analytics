# AI_CONTEXT

## 1) What this repo is / does

- Tennis video analytics pipeline built around Ultralytics YOLO + OpenCV + pandas.
- Primary entrypoint: `src/main.py`.
- Core flow: detect ball -> interpolate trajectory -> optional players/hits/bounces/calibration -> render outputs.
- Ball detections are converted into dense trajectory (`cx,cy,vx,vy,angle`).
- Players are labeled as `near` / `far` using pose detections and x-gating.
- Hits are inferred from trajectory angle change + distance to player wrists/boxes.
- Bounces are inferred from smoothed vertical motion sign flip + peak scoring.
- Optional homography maps ball points to court meters and computes speed.
- Main artifacts are MP4/CSV/PNG/JSON under `--outdir`.

## 2) Repo map (depth 3)

```text
.
├── requirements.txt
├── PROJECT_SUMMARY.md
├── ARCHITECTURE.md
├── AGENTS.md
├── AI_CONTEXT.md
├── scripts/
│   └── debug_frame.py
├── src/
│   ├── main.py
│   ├── script_ultralytics.py
│   ├── ball/
│   │   ├── detect.py
│   │   ├── track.py
│   │   ├── bounce.py
│   │   └── hit.py
│   ├── court/
│   │   ├── auto_calibrate.py
│   │   ├── homography.py
│   │   └── calibrate_click.py
│   ├── player/
│   │   └── pose.py
│   └── viz/
│       └── render.py
└── videos/
    └── .gitkeep
```

## 3) Pipeline flow

1. `src/main.py::parse_args()` loads CLI flags.
2. Resolve/check paths (`_resolve_path`, `_ensure_file`).
3. Ball model + inference (`ball.detect.load_model`, `run_detection`).
4. Save `detecciones_ultralytics.csv`.
5. Interpolate/extrapolate (`ball.track.interpolar_detecciones`) -> `detecciones_interpoladas.csv`.
6. Optional players (`player.pose.detect_players`) -> `players.csv`.
7. Optional hits (`ball.hit.detect_hits`) -> `hits.csv`.
8. Optional bounces (`ball.bounce.detect_bounces`) -> `bounces.csv` + `bounce_best.json`.
9. Optional calibration (`court.auto_calibrate` or manual `court.homography`) -> `H`.
10. Render annotated video (`viz.render.render_video`) -> `output_ultralytics.mp4`.
11. Save plots (`viz.render.save_direction_plots`) -> angle + polar PNGs.
12. If `H`: project to meters (`apply_homography`) -> `detecciones_metros.csv`; optional `court_overlay.png`.

## 4) CLI quick reference

### Minimal run

```bash
python src/main.py --model models/yolo5_last.pt --video videos/federer_murray_trim.mp4 --outdir outputs
```

### Common flags (from `src/main.py`)

- Core:
  - `--model` (default `models/yolo5_last.pt`)
  - `--video` (default `../videos/federer_murray_trim.mp4`)
  - `--outdir` (default `../outputs`)
  - `--extrap` (default `5`)
  - `--no-gui`
- Calibration:
  - `--auto-calibrate`
  - `--calibration-mode {auto,static,manual}` (default `auto`)
  - `--calibration` (default `None`)
  - `--calibration-visuals`
- Bounces:
  - `--detect-bounces`, `--bounce-visuals`, `--bounce-topk` (3)
  - `--bounce-smooth-window` (5)
  - `--bounce-min-frames-between` (8)
  - `--bounce-dy-threshold`/`--bounce-dy-threshold-px` (1.0)
  - `--bounce-score-threshold` (0.2)
  - `--bounce-speed-min` (2.0)
  - `--bounce-min-drop-px` (6.0)
  - `--bounce-exclude-post-hit` (4), `--bounce-exclude-pre-hit` (0)
  - `--bounce-out` (`bounces.csv`), `--bounce-best-out` (`bounce_best.json`)
  - `--bounce-max-gap-to-next-hit` (15)
- Players:
  - `--detect-players`
  - `--draw-players` (default true) / `--no-draw-players`
  - `--pose-model` (`models/yolov8n-pose.pt`)
  - `--player-xgate-left` (0.18), `--player-xgate-right` (0.82)
  - `--player-calibration-margin-px` (20.0)
  - `--player-min-conf` (0.3)
- Hits:
  - `--detect-hits`, `--hit-out` (`hits.csv`), `--hit-visuals`
- Court keypoints:
  - `--court-keypoints`, `--court-keypoints-visuals`
  - `--court-keypoints-model` (`models/keypoints_model.pth`)
  - `--court-keypoints-every` (`0` means infer once on frame 1 and reuse)

## 5) Key modules and ownership

- `src/main.py`: orchestration + file outputs + optional stage toggles.
- `src/ball/detect.py`: YOLO ball inference and raw detection DataFrame.
- `src/ball/track.py`: interpolation/extrapolation and kinematic columns.
- `src/ball/hit.py`: hit scoring from trajectory + player proximity.
- `src/ball/bounce.py`: bounce scoring from smoothed `cy` derivatives.
- `src/player/pose.py`: pose inference, wrist extraction, near/far assignment.
- `src/court/auto_calibrate.py`: auto frame scoring + court corner estimate.
- `src/court/homography.py`: manual calibration load + perspective projection.
- `src/viz/render.py`: video overlays and direction plots.
- `scripts/debug_frame.py`: one-frame overlay debugging + sideline assertion.

## 6) Key data artifacts

### Output files
- `output_ultralytics.mp4`
- `detecciones_ultralytics.csv`
- `detecciones_interpoladas.csv`
- `detecciones_metros.csv` (if homography)
- `players.csv` (if players/hits)
- `hits.csv` (if hits)
- `bounces.csv`, `bounce_best.json` (if bounces)
- `grafico_angulo_por_frame.png`, `rosa_direcciones.png`
- `court_overlay.png` (if calibration overlay available)
- `court_keypoints_frame1.png` (if `--court-keypoints`)

### Important DataFrame schemas
- Ball raw: `frame,x1,y1,x2,y2,confidence`
- Ball interp: `frame,cx,cy,vx,vy,speed,angle,extrapolated`
- Players: `frame,player,x1,y1,x2,y2,wrist_l_x,wrist_l_y,wrist_r_x,wrist_r_y,conf`
- Hits: `frame_hit,player,cx,cy,hit_score`
- Bounces: `frame_bounce,cx,cy,bounce_score[,X_m,Y_m]`

## 7) Known constraints / assumptions

- Pose weights must exist locally (`models/yolov8n-pose.pt`); no auto-download.
- Relative path handling in `main.py` expects repo-root execution for simplest behavior.
- Auto calibration can be rejected when confidence `< 0.4`.
- `frames_raw` keeps all frames in memory (long videos increase RAM usage).
- Hit thresholds are currently code defaults in `ball.hit.detect_hits` (not exposed via CLI).
- Camera/court assumptions are implicit in near/far logic + calibration ordering.

## 8) Common tasks

- **Implement or tune player filtering using court bounds**:
  - edit `src/player/pose.py::select_near_far_people`
  - calibration-aware gate values are injected from `src/main.py::_build_player_selection_cfg`.
- **Add/modify visualization overlays**:
  - edit `src/viz/render.py::render_video`
  - wire new data from `src/main.py` into `render_kwargs`.
- **Change bounce thresholds**:
  - CLI defaults in `src/main.py::parse_args`
  - algorithm behavior in `src/ball/bounce.py::detect_bounces`.
- **Change hit thresholds**:
  - function defaults in `src/ball/hit.py::detect_hits`
  - add CLI plumbing in `src/main.py` if runtime configurability is needed.
- **Add a new event detector**:
  - create module (likely under `src/ball/`)
  - call from `src/main.py`
  - add optional overlay in `src/viz/render.py`
  - define output CSV schema and persistence path.


- Court keypoints model path: `models/keypoints_model.pth` (loaded once on first frame and reused).


- Court keypoints overlay example:
  - `python src/main.py --model models/yolo5_last.pt --video videos/federer_murray_trim.mp4 --outdir outputs --no-gui --court-keypoints --court-keypoints-visuals`
