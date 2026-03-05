# Project Summary

## Purpose of the project
This repository implements a computer-vision tennis analytics pipeline for broadcast video. It is designed to detect and track core match entities (court, ball, players), derive gameplay events, and generate visualized outputs for analysis and review.

## What the pipeline does
At runtime, the main script (`src/main.py`) loads pretrained models, reads video frames, detects court keypoints, detects ball and players, applies lightweight tracking/assignment logic, and writes an annotated output video.

Simple pipeline diagram:

`video → detection → tracking → events → visualization`

## Main outputs
Current primary output from `src/main.py`:

- Annotated MP4 video (default: `outputs/run1.mp4`, or `<outdir>/output_ultralytics.mp4` when using `--outdir`).

Related debug/analysis scripts can also generate image artifacts and help inspect frame-level behavior:

- `scripts/debug_court.py` (court keypoint debug image)
- `scripts/debug_frame.py` (single-frame overlay debug image)

The broader repository also contains modules for event-style analytics and CSV-oriented workflows (ball trajectory, player tracks, hits, bounces), used in legacy/auxiliary flows.

## Example use cases
- Tennis analytics for match breakdown and metric extraction.
- Research and prototyping of sports CV pipelines.
- Coaching analysis with frame-level overlays and event review.
- Broadcast analysis for automated annotation and highlight support.

## Typical run command

```bash
PYTHONPATH=src python src/main.py \
  --video videos/federer_murray_trim.mp4 \
  --court-model models/model_tennis_court_det.pt \
  --ball-model models/yolo5_last.pt \
  --player-model models/yolov8n.pt \
  --output outputs/run1.mp4
```
