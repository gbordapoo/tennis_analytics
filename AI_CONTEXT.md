# AI Context

## Project type
Computer vision sports analytics (tennis).

## Language
Python.

## Libraries used
- OpenCV
- Ultralytics YOLO
- PyTorch
- NumPy
- pandas

## Video resolution / input domain
Broadcast tennis footage (full-court style coverage is expected).

## Important assumptions
- Input videos show a full tennis court.
- Camera is mostly static (broadcast baseline-style angle).
- Pipeline processes frames sequentially.

## Current runtime entrypoint
- `src/main.py`

Primary CLI pattern:

```bash
PYTHONPATH=src python src/main.py \
  --video videos/federer_murray_trim.mp4 \
  --court-model models/model_tennis_court_det.pt \
  --ball-model models/yolo5_last.pt \
  --player-model models/yolov8n.pt \
  --output outputs/run1.mp4
```

Compatibility aliases currently supported:
- `--model` as alias for `--ball-model`
- `--outdir` writes `<outdir>/output_ultralytics.mp4`

## Non-negotiable rules
- Do not change CLI interface unless requested.
- Do not remove existing modules.
- Maintain compatibility with YOLO/model weight files already stored in `models/`.

## Typical debugging tasks
- Court keypoint accuracy/refinement validation.
- Ball trajectory smoothing/tracking continuity.
- Near/far player assignment quality.
- Hit/event detection logic validation in analytics/legacy components.

## Expected analytics artifacts in repository scope
The project scope includes workflows for:
- ball trajectory
- player tracking
- hit detection
- bounce detection
- court keypoints
- visual overlays on video
- CSV-style analysis outputs

Note: current `src/main.py` focuses on annotated video generation; additional analytics/CSV flows exist in legacy and auxiliary modules/scripts.
