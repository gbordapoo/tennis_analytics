from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models


class CourtKeypointsModel(torch.nn.Module):
    """ResNet-style regressor that outputs 14 keypoints (x, y) in 224-space."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(in_features, 28)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_keypoints_model(model_path: str | Path, device: str | None = None) -> torch.nn.Module:
    target = Path(model_path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Court keypoints model not found: {target}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    loaded = torch.load(str(target), map_location=torch_device)
    if isinstance(loaded, torch.nn.Module):
        model = loaded
    else:
        state = loaded
        if isinstance(loaded, dict) and "state_dict" in loaded:
            state = loaded["state_dict"]
        model = CourtKeypointsModel()
        if not isinstance(state, dict):
            raise RuntimeError("Unsupported keypoints checkpoint format")
        model.load_state_dict(state, strict=False)

    model.to(torch_device)
    model.eval()
    return model


def _preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def predict_court_keypoints(frame_bgr: np.ndarray, model: torch.nn.Module) -> list[float]:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty frame provided to keypoint predictor")

    orig_h, orig_w = frame_bgr.shape[:2]
    inputs = _preprocess_frame(frame_bgr).to(next(model.parameters()).device)

    with torch.no_grad():
        pred = model(inputs).detach().cpu().numpy().reshape(-1)

    scale_x = float(orig_w) / 224.0
    scale_y = float(orig_h) / 224.0
    mapped: list[float] = []
    for idx in range(0, len(pred), 2):
        mapped.append(float(pred[idx] * scale_x))
        mapped.append(float(pred[idx + 1] * scale_y))
    return mapped[:28]
