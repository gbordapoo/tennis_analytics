import cv2
import numpy as np
import torch


class CourtKeypointDetector:
    def __init__(self, model_path: str) -> None:
        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()

    def preprocess(self, frame):
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frame = torch.tensor(frame).float().unsqueeze(0)
        return frame

    def predict(self, frame):
        x = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(x)
        keypoints = output.reshape(-1, 2).numpy()
        return keypoints
