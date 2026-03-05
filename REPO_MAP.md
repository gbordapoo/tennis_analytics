# REPO_MAP.md

## Repository Overview

This repository implements a computer vision pipeline for tennis match analysis from broadcast video. It combines court keypoint estimation, ball detection/tracking, player detection/tracking, and event analysis to produce structured tennis analytics outputs.

Technologies used:

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- NumPy
- pandas

Primary goal:

Detect and analyze tennis gameplay events from video.

Key outputs:

- annotated video
- ball trajectory
- player tracking
- hit detection
- bounce detection
- CSV datasets

## Repository Structure

Repository tree (depth ≤ 3):

```text
.
├── AGENTS.md
├── AI_CONTEXT.md
├── ARCHITECTURE.md
├── PROJECT_SUMMARY.md
├── requirements.txt
├── REPO_MAP.md
├── DEBUG_GUIDE.md
├── scripts/
│   ├── debug_court.py
│   ├── debug_frame.py
│   └── diagnose_run.py
├── src/
│   ├── main.py
│   ├── script_ultralytics.py
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── assign.py
│   │   ├── assign_players.py
│   │   ├── event_detector.py
│   │   └── metrics.py
│   ├── ball/
│   │   ├── __init__.py
│   │   ├── bounce.py
│   │   ├── detect.py
│   │   ├── hit.py
│   │   └── track.py
│   ├── court/
│   │   ├── __init__.py
│   │   ├── auto_calibrate.py
│   │   ├── calibrate_click.py
│   │   ├── court_detector.py
│   │   ├── court_geometry.py
│   │   ├── geometry.py
│   │   ├── homography.py
│   │   ├── homography_refine.py
│   │   ├── infer.py
│   │   ├── keypoint_detector.py
│   │   ├── keypoints.py
│   │   ├── model.py
│   │   ├── models.py
│   │   ├── postprocess.py
│   │   ├── stabilize.py
│   │   └── tracknet.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── ball.py
│   │   ├── ball_detector.py
│   │   ├── player_detector.py
│   │   └── players.py
│   ├── player/
│   │   └── pose.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── ball_track.py
│   │   └── object_tracker.py
│   └── viz/
│       ├── __init__.py
│       ├── draw.py
│       └── render.py
└── videos/
    └── .gitkeep
```

## Main Entry Point

Main script:

`src/main.py`

Responsibilities:

- parse CLI arguments
- load models
- process video frames
- run detection modules
- generate outputs

Typical command:

```bash
PYTHONPATH=src python src/main.py \
  --video videos/federer_murray_trim.mp4 \
  --court-model models/model_tennis_court_det.pt \
  --ball-model models/yolo5_last.pt \
  --player-model models/yolov8n.pt \
  --output outputs/run1.mp4
```

## Model Summary

Ball detection

Model:
YOLOv5

Weights:
`models/yolo5_last.pt`

Purpose:
Detect tennis ball positions in each frame.

Player detection

Model:
YOLOv8

Weights:
`models/yolov8n.pt`

Purpose:
Detect players and classify near/far player.

Court detection

Model:
TrackNet-based neural network

Weights:
`models/model_tennis_court_det.pt`

Purpose:
Predict 14 tennis court keypoints.

## Data Flow

Video frame  
↓  
Court detection  
↓  
Ball detection  
↓  
Player detection  
↓  
Trajectory interpolation  
↓  
Hit detection  
↓  
Bounce detection  
↓  
Visualization  
↓  
Output video + CSV files

## Key Scripts

`scripts/debug_court.py`

Used to test the court detector on a single frame.

`scripts/debug_frame.py`

Used to debug detection modules on individual frames.
