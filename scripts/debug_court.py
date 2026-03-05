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


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug TennisCourtDetector on a single frame")
    p.add_argument("--video", required=True)
    p.add_argument("--court-model", default="models/model_tennis_court_det.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--court-refine-lines", type=_parse_bool, default=True)
    p.add_argument("--court-refine-homography", type=_parse_bool, default=True)
    p.add_argument("--court-crop-size", type=int, default=40)
    p.add_argument("--court-max-shift-px", type=float, default=35)
    return p.parse_args()


def _draw_points(img, points, color):
    for idx, (x, y) in enumerate(points):
        if x is None or y is None:
            continue
        p = (int(round(x)), int(round(y)))
        cv2.circle(img, p, 5, color, -1)
        cv2.putText(img, str(idx), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read first frame from {args.video}")

    detector = TennisCourtDetector(
        args.court_model,
        device=args.device,
        refine_lines=args.court_refine_lines,
        refine_homography=args.court_refine_homography,
        crop_size=args.court_crop_size,
        max_shift_px=args.court_max_shift_px,
    )
    final_points = detector.detect(frame)
    raw_points = detector.last_raw_keypoints
    refined_points = detector.last_refined_keypoints

    raw_valid = sum(1 for x, y in raw_points if x is not None and y is not None)
    refined_changed = sum(
        1
        for (rx, ry), (fx, fy) in zip(raw_points, refined_points)
        if rx is not None and ry is not None and fx is not None and fy is not None and (abs(rx - fx) > 1e-3 or abs(ry - fy) > 1e-3)
    )
    homography_replaced = sum(
        1
        for (fx, fy), (tx, ty) in zip(refined_points, final_points)
        if fx is not None and fy is not None and tx is not None and ty is not None and (abs(fx - tx) > 1e-3 or abs(fy - ty) > 1e-3)
    )

    print(f"raw points found: {raw_valid}/{len(raw_points)}")
    print(f"refined points changed: {refined_changed}")
    print(f"points replaced by homography: {homography_replaced}")

    annotated = frame.copy()
    _draw_points(annotated, raw_points, (0, 0, 255))
    _draw_points(annotated, refined_points, (0, 255, 255))
    _draw_points(annotated, final_points, (0, 255, 0))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), annotated)
    print(f"saved debug image: {out}")


if __name__ == "__main__":
    main()
