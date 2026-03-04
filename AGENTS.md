# AGENTS.md

Agent + contributor contract for this repository.

## Canonical run commands

Run from repo root.

- Minimal run:
  - `python src/main.py --model models/best.pt --video videos/federer_murray_trim.mp4 --outdir outputs`
- No-GUI run:
  - `python src/main.py --model models/best.pt --video videos/federer_murray_trim.mp4 --outdir outputs --no-gui`
- Full-features run:
  - `python src/main.py --model models/best.pt --video videos/federer_murray_trim.mp4 --outdir outputs --no-gui --detect-players --detect-hits --hit-visuals --detect-bounces --bounce-visuals --auto-calibrate --calibration-visuals`

Useful tools:
- Manual calibration: `python src/court/calibrate_click.py --video videos/federer_murray_trim.mp4 --output outputs/calibration.json`
- Frame debug: `python scripts/debug_frame.py --video videos/federer_murray_trim.mp4 --frame 58 --outdir outputs --players-csv outputs/players.csv --ball-csv outputs/detecciones_interpoladas.csv`

## Dev setup

- Python version: **Unknown (not found in repo)**.
  - Searched: `requirements.txt`, `pyproject.toml`, `setup.cfg`, `setup.py`, `.python-version`.
- Dependencies in `requirements.txt`:
  - `ultralytics`, `opencv-python`, `pandas`, `numpy`, `matplotlib`, `torch`

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Tests / lint

- Unit tests: **Unknown (not found in repo)** (no `tests/` directory found).
- Lint/format tooling: **Unknown (not found in repo)** (no ruff/flake8/black config found).

Minimum checks before opening a PR:
- `python src/main.py --help` (after CLI changes).
- Run a short `--no-gui` pipeline and confirm expected outputs in `--outdir`.
- If changing near/far filtering, use `scripts/debug_frame.py --assert-far-not-sideline` on sample frames.

## Repo conventions

- Orchestration entrypoint: `src/main.py`.
- Domain modules:
  - `src/ball/` (detect/track/hit/bounce)
  - `src/player/` (pose + near/far assignment)
  - `src/court/` (calibration + homography)
  - `src/viz/` (rendering/plots)
- Data formats (keep stable):
  - ball trajectory: `frame,cx,cy,vx,vy,speed,angle,extrapolated`
  - players: `frame,player,x1,y1,x2,y2,wrist_*,conf`
  - hits: `frame_hit,player,cx,cy,hit_score`
  - bounces: `frame_bounce,cx,cy,bounce_score` (+optional `X_m,Y_m`)
- Output locations:
  - primary outputs in `--outdir`
  - relative `--hit-out`, `--bounce-out`, `--bounce-best-out` are resolved under `--outdir`
- Path behavior:
  - relative paths are resolved from project root first, then `src/` directory (`_resolve_path`).

## Safety rails

### Do
- Keep output schemas backward compatible.
- Keep expensive features behind explicit flags.
- Document any threshold/default changes.

### Don’t
- Don’t change CLI flags/defaults without updating:
  - `PROJECT_SUMMARY.md`
  - `ARCHITECTURE.md` (CLI reference)
  - this `AGENTS.md`
  - `AI_CONTEXT.md`
- Don’t break output filenames/columns without migration notes.
- Don’t add heavy dependencies without explicit justification.
- Don’t re-enable implicit model downloads for pose weights.

## Coupling map (update together)

- If `src/main.py` CLI changes:
  - update all docs (`PROJECT_SUMMARY.md`, `ARCHITECTURE.md`, `AGENTS.md`, `AI_CONTEXT.md`).
- If player selection changes (`src/player/pose.py`):
  - verify hit quality in `src/ball/hit.py` (distance coupling).
  - verify overlay assumptions in `src/viz/render.py` (`near`/`far`).
  - verify debug assertions in `scripts/debug_frame.py` (`xgate`).
- If calibration ordering/contract changes (`src/court/*`):
  - update click-order docs in `src/court/calibrate_click.py` usage guidance.
  - update homography consumers in `src/main.py` and `src/court/homography.py`.
  - verify calibration overlays in `src/viz/render.py` and `draw_court_overlay`.
- If bounce/hit schema changes:
  - update render lookup logic in `src/viz/render.py`.
  - update best-bounce JSON writer in `src/main.py`.
- If trajectory schema changes (`src/ball/track.py`):
  - update `src/viz/render.py`, `src/ball/bounce.py`, and `src/ball/hit.py`.

## Notes for AI agents

- Prefer small, verifiable diffs.
- Keep docs factual and code-grounded.
- If missing info, write: `Unknown (not found in repo)` and list where you searched.
