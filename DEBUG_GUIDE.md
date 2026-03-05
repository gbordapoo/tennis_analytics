# DEBUG_GUIDE.md

## Debugging Strategy

When debugging the pipeline, isolate each component:

1) court detection  
2) ball detection  
3) player detection  
4) event detection

Always debug one module at a time.

## Debug Court Detection

Run:

```bash
PYTHONPATH=src python scripts/debug_court.py \
  --video videos/federer_murray_trim.mp4 \
  --court-model models/model_tennis_court_det.pt
```

Expected output:

- image with 14 court keypoints
- points roughly aligned with court lines

Common issues:

Points clustered at center

Cause:  
incorrect scaling from heatmap to image coordinates

Fix:  
scale coordinates from 640x360 to original frame size.

Points missing

Cause:  
heatmap decoding threshold too high

Fix:  
fallback to argmax when postprocess fails.

Points misaligned

Cause:  
incorrect preprocessing

Check:  
resize to (640,360)  
normalize to [0,1]  
CHW tensor format.

## Debug Ball Detection

Run pipeline with ball detection only.

Verify:

- ball appears in most frames
- trajectory smooths correctly

Common issues:

Ball disappears

Cause:  
confidence threshold too high

Ball jitter

Cause:  
missing interpolation

## Debug Player Detection

Ensure only two players remain.

Common logic:

choose two players closest to baseline positions.

## Debug Event Detection

Events depend on trajectory:

Hit detection

Detect sudden angle change in trajectory.

Bounce detection

Detect vertical direction change.

## Debug Video Rendering

Check overlay:

- ball
- players
- court keypoints
- trajectories

## Debug Workflow

When pipeline fails:

Step 1  
run debug_court.py

Step 2  
verify ball detections

Step 3  
verify player detection

Step 4  
verify trajectory interpolation

Step 5  
verify event detection

## Recommended Debug Order

1 court  
2 ball  
3 players  
4 hits  
5 bounces  
6 visualization

## Useful Commands

Show CLI options:

```bash
PYTHONPATH=src python src/main.py --help
```

Test single frame:

```bash
PYTHONPATH=src python scripts/debug_frame.py
```

Run full pipeline:

```bash
PYTHONPATH=src python src/main.py \
  --video videos/federer_murray_trim.mp4 \
  --court-model models/model_tennis_court_det.pt \
  --ball-model models/yolo5_last.pt \
  --player-model models/yolov8n.pt \
  --output outputs/run1.mp4
```
