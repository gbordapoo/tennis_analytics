from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2

from court.court_detector import TennisCourtDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug TennisCourtDetector on a single frame")
    p.add_argument("--video", required=True)
    p.add_argument("--court-model", default="models/model_tennis_court_det.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read first frame from {args.video}")

    detector = TennisCourtDetector(args.court_model, device=args.device)
    points = detector.predict(frame)

    valid = [(x, y) for x, y in points if x is not None and y is not None]
    if valid:
        xs = [x for x, _ in valid]
        ys = [y for _, y in valid]
        print(
            f"pred points min/max x=({min(xs):.2f},{max(xs):.2f}) "
            f"y=({min(ys):.2f},{max(ys):.2f}) valid={len(valid)}/{len(points)}"
        )
    else:
        print(f"pred points min/max x=(None,None) y=(None,None) valid=0/{len(points)}")

    annotated = detector.draw(frame, points)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), annotated)
    print(f"saved debug image: {out}")


if __name__ == "__main__":
    main()
