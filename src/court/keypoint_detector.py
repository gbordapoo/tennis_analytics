from __future__ import annotations

import numpy as np
import torch

from court.infer import predict_keypoints
from court.model import load_keypoints_model


class CourtKeypointDetector:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        self.model = load_keypoints_model(model_path, device)

        state_dict = torch.load(model_path, map_location="cpu")

        # if checkpoint stored as dict with key 'state_dict'
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # remove DataParallel prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k.replace("module.", "")] = v
            else:
                new_state_dict[k] = v

        state_dict = new_state_dict

        # load weights with compatibility report
        res = self.model.load_state_dict(state_dict, strict=False)

        print("\n--- COURT MODEL LOAD REPORT ---")
        print("missing keys:", len(res.missing_keys))
        print("unexpected keys:", len(res.unexpected_keys))
        print("example missing:", res.missing_keys[:10])
        print("example unexpected:", res.unexpected_keys[:10])
        print("--------------------------------\n")

        self.model.eval()

    def predict(self, frame: np.ndarray, scale_to_frame: bool = True) -> np.ndarray:
        kpts = predict_keypoints(self.model, frame, self.device)
        if not scale_to_frame:
            return kpts
        return kpts.astype(np.int32)
