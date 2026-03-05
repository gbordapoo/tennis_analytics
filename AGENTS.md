# AGENTS.md

## Repository purpose
Computer-vision tennis analytics pipeline for broadcast video: court/ball/player detection, tracking, event-oriented analysis, and visual output generation.

## Quick start commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run pipeline CLI help:

```bash
PYTHONPATH=src python src/main.py --help
```

Run pipeline example:

```bash
PYTHONPATH=src python src/main.py \
  --video videos/federer_murray_trim.mp4 \
  --court-model models/model_tennis_court_det.pt \
  --ball-model models/yolo5_last.pt \
  --player-model models/yolov8n.pt \
  --output outputs/run1.mp4
```

## Debug commands

```bash
PYTHONPATH=src python scripts/debug_court.py --video videos/federer_murray_trim.mp4 --out outputs/debug_court.png
PYTHONPATH=src python scripts/debug_frame.py --video videos/federer_murray_trim.mp4 --frame 58 --outdir outputs
```

## Coding guidelines
- Prefer small, targeted changes.
- Do not refactor entire modules unless requested.
- Always preserve CLI compatibility.
- Keep code Pythonic and readable.
- Keep docs synchronized with `src/main.py` behavior.

## When modifying models
- Never retrain models automatically.
- Assume pretrained weights already exist in `/models`.
- Do not introduce implicit model downloads.
