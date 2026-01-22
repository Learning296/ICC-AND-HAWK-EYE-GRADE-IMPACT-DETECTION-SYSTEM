import cv2
import torch
import numpy as np
import torchvision.models as models

class ImpactDetector:
    def __init__(self, weight_path, threshold=0.7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -------- MODEL (MATCHING YOUR WEIGHTS) --------
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

        state = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.threshold = threshold
        self.size = 224

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.size, self.size))
        img = img[:, :, ::-1]  # BGR → RGB
        img = img.astype(np.float32) / 255.0

        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)

    def predict(self, frame):
        x = self.preprocess(frame)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

        impact_prob = probs[0, 1].item()
        return impact_prob > self.threshold, impact_prob
