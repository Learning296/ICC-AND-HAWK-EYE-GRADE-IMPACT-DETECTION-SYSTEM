import cv2
import torch
import numpy as np
from torchvision.models.video import r3d_18

class ImpactDetector:
    def __init__(self, weight_path, threshold=0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load 3D ResNet model
        self.model = r3d_18(weights=None)
        self.model.fc = torch.nn.Linear(512, 1)

        state = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.threshold = threshold
        self.num_frames = 16
        self.size = 112

    def extract_clip(self, frames, center_idx, fps):
        half = int(0.5 * fps)
        start = max(0, center_idx - half)
        end = min(len(frames), center_idx + half)
        return frames[start:end]

    def preprocess(self, clip):
        if len(clip) < self.num_frames:
            clip += [clip[-1]] * (self.num_frames - len(clip))

        idxs = np.linspace(0, len(clip) - 1, self.num_frames).astype(int)
        frames = []

        for i in idxs:
            img = cv2.resize(clip[i], (self.size, self.size))
            img = img[:, :, ::-1]  # BGR → RGB
            img = img.astype(np.float32) / 255.0
            frames.append(img)

        x = np.stack(frames)                 # T,H,W,C
        x = torch.tensor(x).permute(3, 0, 1, 2)  # C,T,H,W
        x = x.unsqueeze(0).to(self.device)   # 1,C,T,H,W
        return x

    def predict(self, clip):
        x = self.preprocess(clip)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits)[0, 0].item()

        return prob > self.threshold, prob
