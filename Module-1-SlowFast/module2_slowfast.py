import cv2
import torch
import numpy as np
from pytorchvideo.models.hub import slowfast_r50

class SlowFastImpactDetector:
    def __init__(self, weight_path, threshold=0.7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = slowfast_r50(pretrained=False)
        self.model.blocks[-1].proj = torch.nn.Linear(
            self.model.blocks[-1].proj.in_features, 2
        )

        state = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.threshold = threshold
        self.num_frames = 32
        self.size = 224

    def extract_clip(self, frames, center_idx, fps):
        half = int(1.0 * fps)
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
            img = img.astype(np.float32) / 255.0
            frames.append(img)

        x = np.stack(frames)                 # T,H,W,C
        x = torch.tensor(x).permute(3, 0, 1, 2)  # C,T,H,W
        x = x.unsqueeze(0).to(self.device)

        return [x[:, :, ::4], x]  # SlowFast dual pathway

    def predict(self, clip):
        if len(clip) < self.num_frames:
            return False, 0.0

        inputs = self.preprocess(clip)

        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)

        impact_prob = probs[0, 1].item()
        return impact_prob > self.threshold, impact_prob
