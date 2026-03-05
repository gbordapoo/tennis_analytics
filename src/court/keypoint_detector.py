import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np


class CourtKeypointDetector:

    def __init__(self, model_path):

        # create ResNet50 architecture
        self.model = models.resnet50(weights=None)

        # replace final layer
        self.model.fc = nn.Linear(self.model.fc.in_features, 28)

        # load weights (state_dict)
        state_dict = torch.load(model_path, map_location="cpu")

        self.model.load_state_dict(state_dict)

        self.model.eval()
        print("Court keypoint model loaded:", model_path)

    def preprocess(self, frame):

        img = cv2.resize(frame, (224,224))
        img = img.astype("float32") / 255.0

        img = np.transpose(img, (2,0,1))

        img = torch.tensor(img).unsqueeze(0)

        return img

    def predict(self, frame):

        input_tensor = self.preprocess(frame)

        with torch.no_grad():
            output = self.model(input_tensor)

        keypoints = output.numpy().reshape(-1,2)

        return keypoints
