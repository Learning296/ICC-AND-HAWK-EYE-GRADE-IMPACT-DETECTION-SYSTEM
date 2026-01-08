import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models.video import r3d_18


class SlowFastR3D:
    def __init__(self, threshold=0.3, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # -------- SETTINGS --------
        self.num_frames_fast = 16
        self.num_frames_slow = 4
        self.size = 112

        # -------- MODELS --------
        self.fast_model = r3d_18(weights=None)
        self.slow_model = r3d_18(weights=None)

        self.fast_model.fc = nn.Identity()
        self.slow_model.fc = nn.Identity()

        self.fast_model.to(self.device).eval()
        self.slow_model.to(self.device).eval()

        self.classifier = nn.Linear(512 * 2, 1).to(self.device)

    # -------- CLIP EXTRACTION --------
    def extract_clip(self, frames, center_idx, fps):
        half = int(0.5 * fps)
        start = max(0, center_idx - half)
        end = min(len(frames), center_idx + half)
        return frames[start:end]

    # -------- PREPROCESS --------
    def preprocess(self, clip):
        frames = [
            cv2.resize(f, (self.size, self.size))[:, :, ::-1].astype(np.float32) / 255.0
            for f in clip
        ]

        # Fast path
        idx_fast = np.linspace(0, len(frames) - 1, self.num_frames_fast).astype(int)
        fast = np.stack([frames[i] for i in idx_fast])
        fast = torch.tensor(fast).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)

        # Slow path (subsample fast)
        idx_slow = np.linspace(0, self.num_frames_fast - 1, self.num_frames_slow).astype(int)
        slow = fast[:, :, idx_slow, :, :]

        return slow, fast

    # -------- PREDICT (CLIP LEVEL) --------
    def predict(self, clip):
        slow, fast = self.preprocess(clip)

        with torch.no_grad():
            fs = self.slow_model(slow)
            ff = self.fast_model(fast)
            fused = torch.cat([fs, ff], dim=1)
            logit = self.classifier(fused)
            prob = torch.sigmoid(logit)[0, 0].item()

        del slow, fast, fs, ff, fused, logit
        torch.cuda.empty_cache()

        return prob
