from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "net", "model")


class _ModelWithInputSize(torch.nn.Module):
    """Adapter to carry optional input size metadata from checkpoints."""

    def __init__(self, model: torch.nn.Module, input_size: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.model = model
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _detect_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_eval_model(model: torch.nn.Module, torch_device: torch.device) -> torch.nn.Module:
    model.to(torch_device)
    model.eval()
    return model


def _extract_state_dict_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
    state_dict = None
    for key in _STATE_DICT_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break

    raw_size = payload.get("input_size") or payload.get("img_size") or payload.get("image_size")
    input_size: tuple[int, int] | None = None
    if isinstance(raw_size, (tuple, list)) and len(raw_size) >= 2:
        input_size = (int(raw_size[0]), int(raw_size[1]))
    elif isinstance(raw_size, int):
        input_size = (int(raw_size), int(raw_size))

    return state_dict, input_size


def _find_architecture_model() -> torch.nn.Module | None:
    try:
        from court.models import CourtKeypointsModel  # type: ignore

        return CourtKeypointsModel()
    except Exception:
        return None


def load_keypoints_model(model_path: str, device: str | None = None) -> torch.nn.Module:
    target = Path(model_path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Court keypoints model not found: {target}")

    torch_device = _detect_device(device)

    try:
        model = torch.jit.load(str(target), map_location=torch_device)
        return _as_eval_model(model, torch_device)
    except Exception as jit_exc:
        fallback_error = jit_exc

    loaded = torch.load(str(target), map_location=torch_device)

    if isinstance(loaded, torch.nn.Module):
        return _as_eval_model(loaded, torch_device)

    if isinstance(loaded, dict):
        state_dict, input_size = _extract_state_dict_payload(loaded)
        if state_dict is not None:
            arch_model = _find_architecture_model()
            if arch_model is None:
                keys = ", ".join(sorted(loaded.keys()))
                raise RuntimeError(
                    "Unsupported keypoints checkpoint format: state_dict-like payload found but no architecture "
                    "implementation was discovered. Supported formats are TorchScript and pickled nn.Module. "
                    f"Checkpoint keys: [{keys}]"
                ) from fallback_error
            missing = arch_model.load_state_dict(state_dict, strict=False)
            if missing.missing_keys or missing.unexpected_keys:
                print(
                    "⚠️ Loaded keypoints state_dict with non-strict matching "
                    f"(missing={len(missing.missing_keys)}, unexpected={len(missing.unexpected_keys)})."
                )
            wrapped = _ModelWithInputSize(arch_model, input_size=input_size)
            return _as_eval_model(wrapped, torch_device)

    raise RuntimeError(
        "Unsupported keypoints model format. Expected TorchScript, nn.Module checkpoint, or a checkpoint dict "
        "with state_dict/model_state_dict and known architecture code."
    )


def _infer_input_size(model: torch.nn.Module) -> tuple[int, int]:
    size = getattr(model, "input_size", None)
    if isinstance(size, (tuple, list)) and len(size) >= 2:
        return int(size[0]), int(size[1])
    return 224, 224


def _preprocess_frame(frame_bgr: np.ndarray, input_size: tuple[int, int]) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, input_size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def _extract_output_array(output: Any) -> np.ndarray:
    if isinstance(output, (list, tuple)) and output:
        output = output[0]

    if isinstance(output, dict):
        for key in ("keypoints", "pred", "output", "coords"):
            if key in output:
                output = output[key]
                break

    if isinstance(output, torch.Tensor):
        arr = output.detach().cpu().numpy()
    else:
        arr = np.asarray(output)

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    return arr


def _to_kx2(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError(f"Invalid keypoints output size {arr.size}; expected even-length flat vector")
        return arr.reshape(-1, 2)

    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            return arr[:, :2]
        if arr.shape[0] == 2:
            return arr.T

    if arr.ndim == 3:
        squeezed = np.squeeze(arr)
        return _to_kx2(squeezed)

    raise ValueError(f"Unsupported keypoints output shape: {arr.shape}")


def predict_court_keypoints(model: torch.nn.Module, frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty frame provided to keypoint predictor")

    input_w, input_h = _infer_input_size(model)
    inputs = _preprocess_frame(frame_bgr, (input_w, input_h))

    first_param = next(model.parameters(), None)
    model_device = first_param.device if first_param is not None else torch.device("cpu")
    inputs = inputs.to(model_device)

    with torch.no_grad():
        output = model(inputs)

    arr = _extract_output_array(output)
    kps = _to_kx2(arr).astype(np.float32)

    h, w = frame_bgr.shape[:2]
    if np.nanmax(np.abs(kps)) <= 1.5:
        kps[:, 0] *= float(w)
        kps[:, 1] *= float(h)
    elif np.nanmax(kps[:, 0]) <= float(input_w) * 1.25 and np.nanmax(kps[:, 1]) <= float(input_h) * 1.25:
        kps[:, 0] *= float(w) / float(input_w)
        kps[:, 1] *= float(h) / float(input_h)

    return kps.astype(np.float32)


def smooth_or_cache_keypoints(
    cache: dict[int, np.ndarray],
    frame_idx: int,
    keypoints: np.ndarray | None,
    every_n: int = 0,
) -> np.ndarray | None:
    if keypoints is not None:
        cache[int(frame_idx)] = np.asarray(keypoints, dtype=np.float32)
        return cache[int(frame_idx)]

    if not cache:
        return None

    if every_n <= 0:
        first_frame = min(cache.keys())
        return cache[first_frame]

    valid_frames = [f for f in cache.keys() if f <= int(frame_idx)]
    if not valid_frames:
        return None
    return cache[max(valid_frames)]
