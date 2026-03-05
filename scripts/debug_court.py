from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import torch

from court.keypoints import _to_kx2, load_keypoints_model, predict_court_keypoints
from viz.render import render_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug court keypoint decoding on a chosen frame")
    p.add_argument("--video", required=True)
    p.add_argument("--court-model", default="models/keypoints_model.pth")
    p.add_argument("--out", required=True)
    p.add_argument("--frame-idx", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {args.frame_idx} from {args.video}")

    model = load_keypoints_model(args.court_model, device=args.device)
    model_device = next(model.parameters(), torch.empty(0)).device
    with torch.no_grad():
        raw = model(torch.from_numpy(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(model_device))
    raw_kps = _to_kx2(raw)

    decoded = predict_court_keypoints(model, frame)

    print(
        f"raw min/max x=({raw_kps[:,0].min():.4f},{raw_kps[:,0].max():.4f}) "
        f"y=({raw_kps[:,1].min():.4f},{raw_kps[:,1].max():.4f})"
    )
    print(
        f"pixel min/max x=({decoded[:,0].min():.2f},{decoded[:,0].max():.2f}) "
        f"y=({decoded[:,1].min():.2f},{decoded[:,1].max():.2f})"
    )

    overlay = render_frame(frame, decoded, None, None, None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)
    print(f"saved debug image: {out}")


if __name__ == "__main__":
    main()
