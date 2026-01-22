import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader

# ---------------- SETTINGS ----------------
VIDEO_PATH = "videos/video1.mp4"  # inference ke liye
DATASET_DIR = "dataset"
CLIP_LEN = 16
IMG_SIZE = 112
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
class ClipDataset(Dataset):
    def __init__(self, root, clip_len=16, img_size=112):
        self.clips = []
        self.labels = []
        self.clip_len = clip_len
        self.img_size = img_size

        for label, name in [(1,"impact"), (0,"no_impact")]:
            p = os.path.join(root, name)
            for file in os.listdir(p):
                if file.endswith(".mp4") or file.endswith(".MP4"):
                    self.clips.append(os.path.join(p, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        path = self.clips[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)
        cap.release()

        if len(frames) < self.clip_len:
            frames += [frames[-1]] * (self.clip_len - len(frames))

        idxs = np.linspace(0, len(frames)-1, self.clip_len).astype(int)
        frames = np.stack([frames[i] for i in idxs]).astype(np.float32)/255.0
        frames = torch.tensor(frames).permute(3,0,1,2)  # C,T,H,W
        return frames, torch.tensor(label, dtype=torch.float32)

# ---------------- LOAD DATA ----------------
ds = ClipDataset(DATASET_DIR, clip_len=CLIP_LEN, img_size=IMG_SIZE)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# ---------------- MODEL ----------------
model = models.video.r3d_18(weights=models.video.R3D_18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)  # binary classification
model = model.to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAINING ----------------
for e in range(EPOCHS):
    print(f"\nEpoch {e+1}/{EPOCHS}")
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x).squeeze()
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        print(f"Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "r3d_impact.pth")
print("✅ TRAINING DONE")

# ---------------- INFERENCE ----------------
def predict_video(model, video_path, clip_len=CLIP_LEN, img_size=IMG_SIZE):
    model.eval()
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()

    if len(frames) < clip_len:
        frames += [frames[-1]] * (clip_len - len(frames))

    clip_idxs = np.arange(0, len(frames) - clip_len + 1, clip_len)
    probs = []

    with torch.no_grad():
        for start in clip_idxs:
            clip = [frames[i] for i in range(start, start+clip_len)]
            clip = np.stack(clip).astype(np.float32)/255.0
            clip = torch.tensor(clip).permute(3,0,1,2).unsqueeze(0).to(DEVICE)  # 1,C,T,H,W
            out = model(clip).squeeze()
            prob = torch.sigmoid(out).item()
            probs.append(prob)
            print(f"Impact probability: {prob:.3f}")

    return probs

# Example usage:
# probs = predict_video(model, VIDEO_PATH)
