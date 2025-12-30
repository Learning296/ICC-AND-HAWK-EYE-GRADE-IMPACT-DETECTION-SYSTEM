# module2_slowfast.py
import cv2
import torch
import numpy as np
from pytorchvideo.models.hub import x3d_m

class SlowFastImpactDetector:
    def __init__(self, threshold=0.7, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = x3d_m(pretrained=True)
        self.model.blocks[-1].proj = torch.nn.Linear(2048, 2)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.threshold = threshold
        self.num_frames = 16
        self.size = 224   # 🔥 FIX HERE

    def extract_clip(self, frames, center_idx, fps):
        half = int(1.5 * fps)
        start = max(0, center_idx - half)
        end = min(len(frames), center_idx + half)
        return frames[start:end]

    def preprocess(self, clip):
        idxs = np.linspace(0, len(clip)-1, self.num_frames).astype(int)
        sampled = []

        for i in idxs:
            img = cv2.resize(clip[i], (self.size, self.size))
            sampled.append(img)

        clip = np.stack(sampled).astype(np.float32) / 255.0
        clip = torch.tensor(clip).permute(3, 0, 1, 2)  # C,T,H,W
        clip = clip.unsqueeze(0).to(self.device)      # B,C,T,H,W
        return clip

    def confirm_impact(self, clip):
        if len(clip) < self.num_frames:
            return False, 0.0

        x = self.preprocess(clip)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

        impact_prob = probs[0, 1].item()
        return impact_prob > self.threshold, impact_prob
