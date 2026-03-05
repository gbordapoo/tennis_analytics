from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224


def preprocess(frame: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)


def infer_output_domain(coords: np.ndarray) -> str:
    min_v = float(np.nanmin(coords))
    max_v = float(np.nanmax(coords))
    if min_v >= -0.05 and max_v <= 1.05:
        return "normalized"
    if min_v >= -5.0 and max_v <= INPUT_SIZE + 5.0:
        return "resized"
    return "pixel"


def predict_keypoints(model: torch.nn.Module, frame: np.ndarray, device: str) -> np.ndarray:
    h, w = frame.shape[:2]
    x = preprocess(frame).to(device)
    with torch.no_grad():
        out = model(x)

    coords = out.detach().cpu().numpy().reshape(-1, 2).astype(np.float32)
    domain = infer_output_domain(coords)

    if domain == "normalized":
        x_resized = coords[:, 0] * INPUT_SIZE
        y_resized = coords[:, 1] * INPUT_SIZE
        coords[:, 0] = x_resized * (w / INPUT_SIZE)
        coords[:, 1] = y_resized * (h / INPUT_SIZE)
    elif domain == "resized":
        coords[:, 0] = coords[:, 0] * (w / INPUT_SIZE)
        coords[:, 1] = coords[:, 1] * (h / INPUT_SIZE)

    coords[:, 0] = np.clip(coords[:, 0], 0, max(0, w - 1))
    coords[:, 1] = np.clip(coords[:, 1], 0, max(0, h - 1))
    return coords.astype(np.float32)
