from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class CourtKeypointDetector:
    def __init__(self, model_path: str) -> None:
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 28)

        state_dict = self._extract_state_dict(torch.load(model_path, map_location="cpu"))
        cleaned_state = self._strip_module_prefix(state_dict)

        try:
            self.model.load_state_dict(cleaned_state, strict=True)
        except RuntimeError:
            missing = self.model.load_state_dict(cleaned_state, strict=False)
            print(
                "⚠️ Court keypoint model loaded non-strictly "
                f"(missing={len(missing.missing_keys)}, unexpected={len(missing.unexpected_keys)})"
            )

        self.model.eval()
        print("Court keypoint model loaded:", model_path)

    @staticmethod
    def _extract_state_dict(ckpt: object) -> OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor]:
        if isinstance(ckpt, OrderedDict):
            return ckpt

        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                return ckpt["state_dict"]
            if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                return ckpt["model_state_dict"]
            if "model" in ckpt and isinstance(ckpt["model"], OrderedDict):
                return ckpt["model"]
            if all(isinstance(k, str) for k in ckpt.keys()):
                return ckpt

        raise RuntimeError("Unsupported keypoint checkpoint format; could not extract state_dict")

    @staticmethod
    def _strip_module_prefix(state: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            new_key = key[7:] if key.startswith("module.") else key
            cleaned[new_key] = value
        return cleaned

    @staticmethod
    def preprocess(frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        img = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)
        return tensor

    def predict(self, frame: np.ndarray, scale_to_frame: bool = True) -> np.ndarray:
        frame_h, frame_w = frame.shape[:2]
        input_tensor = self.preprocess(frame)

        with torch.no_grad():
            output = self.model(input_tensor)

        keypoints = output.detach().cpu().numpy().reshape(14, 2)

        if scale_to_frame:
            max_value = float(np.nanmax(np.abs(keypoints)))
            if max_value <= 1.5:
                keypoints[:, 0] *= float(frame_w)
                keypoints[:, 1] *= float(frame_h)
            else:
                keypoints[:, 0] *= float(frame_w) / 224.0
                keypoints[:, 1] *= float(frame_h) / 224.0

            keypoints[:, 0] = np.clip(keypoints[:, 0], 0, frame_w - 1)
            keypoints[:, 1] = np.clip(keypoints[:, 1], 0, frame_h - 1)

        return keypoints.astype(np.int32)
