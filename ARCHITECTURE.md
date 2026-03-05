# Architecture

## Main entrypoint
The primary orchestration entrypoint is:

- `src/main.py`

`src/main.py` coordinates model loading, frame iteration, court keypoint detection, ball/player detection, near/far player assignment, ball center tracking, visualization, and output video writing.

## Module layout and responsibilities

Requested conceptual modules:

```text
src/
  ball/
  court/
  players/
  hits/
  bounces/
  viz/
```

Current repository implementation (actual paths) is:

```text
src/
  ball/        # legacy ball/event components (detect/track/hit/bounce)
  court/       # court keypoint model, postprocess, geometry, homography tools
  player/      # legacy pose/player logic
  detection/   # current runtime ball + player detectors used by src/main.py
  tracking/    # current runtime trackers used by src/main.py
  analytics/   # player assignment + simple event/metric helpers
  viz/         # rendering overlays
```

Responsibilities by area:

- `src/ball/`: trajectory and event logic from the legacy pipeline (`detect.py`, `track.py`, `hit.py`, `bounce.py`).
- `src/court/`: court keypoint inference and refinement (`court_detector.py`, `tracknet.py`, postprocessing/homography helpers).
- `src/player/`: legacy player/pose-oriented utilities.
- `src/detection/`: runtime detectors currently invoked by `src/main.py`:
  - `ball.py` (`BallDetector` with YOLO model)
  - `players.py` (`PlayerDetector` with YOLO model)
- `src/tracking/`: runtime tracking helpers (`SimpleBallTracker`, `SimpleCentroidTracker`).
- `src/analytics/`: near/far player assignment plus lightweight analytics helpers.
- `src/viz/`: frame rendering and overlays (`render.py`, `draw.py`).

## Data flow

Typical frame-level flow in current runtime:

`video frame`
`→ court detection`
`→ ball detection`
`→ player detection`
`→ trajectory interpolation / temporal tracking`
`→ event detection / assignment`
`→ visualization`
`→ CSV export (legacy/auxiliary flows) or video export (current main flow)`

In `src/main.py`, the current implemented output path is annotated video export; CSV/event exports are represented in other modules and prior workflows.

## Key models

- Ball detector:
  - YOLOv5-style weights file: `models/yolo5_last.pt`
  - Used by: `src/detection/ball.py`

- Player detector:
  - YOLOv8 weights file: `models/yolov8n.pt`
  - Used by: `src/detection/players.py`

- Court keypoints:
  - TrackNet-style keypoint model loaded via `TennisCourtDetector`
  - Typical weights file: `models/model_tennis_court_det.pt`
  - Used by: `src/court/court_detector.py`

## Debugging utilities

- `scripts/debug_court.py`
  - Runs court detection on the first frame and saves annotated keypoint diagnostics.
  - Useful for validating keypoint confidence/refinement behavior.

- `scripts/debug_frame.py`
  - Renders a chosen frame with player/ball overlays from CSV inputs.
  - Useful for near/far assignment checks and frame-specific visual debugging.
