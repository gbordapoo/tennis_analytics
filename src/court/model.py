from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models


class CourtKeypointNet(nn.Module):
    def __init__(self, num_keypoints: int = 14, backbone: str = "resnet18") -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_keypoints * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _extract_state_dict(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], (dict, OrderedDict)):
            return dict(ckpt["state_dict"])
        if "model" in ckpt and isinstance(ckpt["model"], (dict, OrderedDict)):
            return dict(ckpt["model"])
        if all(isinstance(k, str) for k in ckpt.keys()):
            return dict(ckpt)

    if isinstance(ckpt, OrderedDict):
        return dict(ckpt)

    raise RuntimeError("Unsupported keypoint checkpoint format.")


def load_keypoints_model(path: str, device: str, num_keypoints: int = 14) -> nn.Module:
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        model = CourtKeypointNet(num_keypoints=num_keypoints)
        state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model
