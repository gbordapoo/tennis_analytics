from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from court.infer import preprocess, predict_keypoints
from court.model import load_keypoints_model
from viz.draw import draw_keypoints


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--court-model", default="models/keypoints_model.pth")
    p.add_argument("--out", default="outputs/debug_court.png")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Unable to read first frame")

    model = load_keypoints_model(args.court_model, args.device)
    raw = model(preprocess(frame).to(args.device)).detach().cpu().numpy().reshape(-1, 2)
    final_kpts = predict_keypoints(model, frame, args.device)

    print(f"raw min/max: {raw.min():.4f} / {raw.max():.4f}")
    print(f"pixel min/max: {final_kpts.min():.2f} / {final_kpts.max():.2f}")

    draw_keypoints(frame, final_kpts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
