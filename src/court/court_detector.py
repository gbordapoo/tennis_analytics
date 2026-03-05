import cv2
import numpy as np
import torch

from .tracknet import BallTrackerNet


class TennisCourtDetector:
    def __init__(self, model_path, device="cpu"):
        self.device = device

        self.model = BallTrackerNet(out_channels=15)

        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        state_dict = {}

        for k, v in checkpoint.items():
            state_dict[k.replace("module.", "")] = v

        self.model.load_state_dict(state_dict)

        self.model.to(device)
        self.model.eval()

        self.input_w = 640
        self.input_h = 360

    def predict(self, frame):
        H, W, _ = frame.shape

        img = cv2.resize(frame, (self.input_w, self.input_h))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))

        tensor = torch.tensor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        out = torch.sigmoid(out)

        heatmaps = out[0].cpu().numpy()

        keypoints = []

        for i in range(14):
            heat = heatmaps[i]

            idx = heat.argmax()

            y, x = np.unravel_index(idx, heat.shape)

            x = x * (self.input_w / heat.shape[1])
            y = y * (self.input_h / heat.shape[0])

            x = x * (W / self.input_w)
            y = y * (H / self.input_h)

            keypoints.append((x, y))

        return np.array(keypoints)
