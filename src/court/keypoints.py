from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from court.models import CourtKeypointsModel

_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "net", "model")


def _device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_eval(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model = model.to(device)
    model.eval()
    return model


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(payload, OrderedDict):
        return dict(payload)
    if isinstance(payload, dict):
        for key in _STATE_DICT_KEYS:
            v = payload.get(key)
            if isinstance(v, (dict, OrderedDict)):
                return dict(v)
    return None


def load_keypoints_model(model_path: str, device: str | None = None) -> torch.nn.Module:
    target = Path(model_path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Court keypoints model not found: {target}")

    torch_device = _device(device)

    try:
        scripted = torch.jit.load(str(target), map_location=torch_device)
        return _to_eval(scripted, torch_device)
    except Exception:
        pass

    loaded = torch.load(str(target), map_location=torch_device)
    if isinstance(loaded, torch.nn.Module):
        return _to_eval(loaded, torch_device)

    state_dict = _extract_state_dict(loaded)
    if state_dict is None:
        raise RuntimeError(
            "Unsupported keypoints model format. Use TorchScript or nn.Module checkpoint."
        )

    model = CourtKeypointsModel()
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            "⚠️ Keypoint state_dict loaded with strict=False "
            f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
        )

    if isinstance(loaded, OrderedDict):
        # explicit message for classic failure mode
        print(
            "⚠️ Loaded OrderedDict checkpoint into CourtKeypointsModel. "
            "Prefer exporting TorchScript for portable inference."
        )

    return _to_eval(model, torch_device)


def _preprocess(frame_bgr: np.ndarray, input_w: int = 224, input_h: int = 224) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return x


def _to_kx2(output: Any) -> np.ndarray:
    if isinstance(output, (tuple, list)) and output:
        output = output[0]
    if isinstance(output, dict):
        for key in ("keypoints", "pred", "output", "coords"):
            if key in output:
                output = output[key]
                break

    arr = output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else np.asarray(output)
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError(f"Invalid keypoint output shape: {arr.shape}")
        return arr.reshape(-1, 2)
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr
        if arr.shape[0] == 2:
            return arr.T
    raise ValueError(f"Unsupported keypoint output shape: {arr.shape}")


def predict_court_keypoints(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    previous_keypoints: np.ndarray | None = None,
    input_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    input_w, input_h = input_size
    tensor = _preprocess(frame_bgr, input_w=input_w, input_h=input_h)
    model_device = next(model.parameters(), torch.empty(0)).device
    with torch.no_grad():
        pred = model(tensor.to(model_device))

    kps = _to_kx2(pred)
    xmin, xmax = float(np.min(kps[:, 0])), float(np.max(kps[:, 0]))
    ymin, ymax = float(np.min(kps[:, 1])), float(np.max(kps[:, 1]))

    h, w = frame_bgr.shape[:2]
    if xmin >= 0 and xmax <= 1.5 and ymin >= 0 and ymax <= 1.5:
        x_px = kps[:, 0] * float(w)
        y_px = kps[:, 1] * float(h)
    elif xmin >= -2.5 and xmax <= 2.5 and ymin >= -2.5 and ymax <= 2.5:
        x01 = np.clip((kps[:, 0] + 1.0) * 0.5, 0.0, 1.0)
        y01 = np.clip((kps[:, 1] + 1.0) * 0.5, 0.0, 1.0)
        x_px = x01 * float(w)
        y_px = y01 * float(h)
    elif xmax <= input_w * 1.25 and ymax <= input_h * 1.25:
        x_px = kps[:, 0] * (float(w) / float(input_w))
        y_px = kps[:, 1] * (float(h) / float(input_h))
    else:
        x_px, y_px = kps[:, 0], kps[:, 1]

    decoded = np.stack([x_px, y_px], axis=1).astype(np.float32)
    if not np.all(np.isfinite(decoded)):
        if previous_keypoints is not None and np.all(np.isfinite(previous_keypoints)):
            return previous_keypoints.astype(np.float32)
        decoded = np.nan_to_num(decoded, nan=0.0, posinf=float(w - 1), neginf=0.0)

    return decoded.astype(np.float32)
