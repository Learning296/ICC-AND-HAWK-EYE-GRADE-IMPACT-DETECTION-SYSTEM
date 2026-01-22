import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ultralytics import YOLO
from torchvision import models

# ---------------- SETTINGS ----------------
VIDEO_PATH = "videos/video2.mp4"  # input video
OUTPUT_PATH = "output_final.mp4"
CLIP_LEN = 16
IMG_SIZE = 112
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Physics-based thresholds
IMPACT_DIST = 40
VELOCITY_THRESHOLD = 3
ACCEL_THRESHOLD = 2.5
PIXEL_DIFF_THRESHOLD = 25
FLOW_PATCH_SIZE = 40
IMPACT_SCORE_THRESHOLD = 0.6
IMPACT_COOLDOWN = 5

# ---------------- LOAD YOLO ----------------
bat_model = YOLO("runs/detect/train/weights/bat_new_2.pt")
ball_model = YOLO("runs/detect/train/weights/besst.pt")

# ---------------- LOAD R3D MODEL ----------------
model = models.video.r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("slowfast_impact.pth"))
model = model.to(DEVICE)
model.eval()

# ---------------- KALMAN FILTER ----------------
class BallKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4,2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32)*0.03
        self.initialized = False
    def update(self, cx=None, cy=None):
        if cx is not None and cy is not None:
            measurement = np.array([[cx],[cy]], np.float32)
            if not self.initialized:
                self.kf.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
                self.initialized = True
            self.kf.correct(measurement)
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

# ---------------- UTILITIES ----------------
def euclidean(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def preprocess_clip(frames):
    if len(frames) < CLIP_LEN:
        frames += [frames[-1]]*(CLIP_LEN - len(frames))
    idxs = np.linspace(0, len(frames)-1, CLIP_LEN).astype(int)
    frames = np.stack([frames[i] for i in idxs]).astype(np.float32)/255.0
    frames = torch.tensor(frames).permute(3,0,1,2).unsqueeze(0).to(DEVICE)
    return frames

# ---------------- MAIN PIPELINE ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, f = cap.read()
    if not ret: break
    frames.append(f)
cap.release()
print(f"Total frames: {len(frames)}")

h,w,_ = frames[0].shape
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w,h))

kalman = BallKalman()
prev_ball = None
prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
ball_history = deque(maxlen=5)
last_impact = -999

impact_frames_detected = []

for i, frame in enumerate(frames):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()
    bat_centers, ball_center = [], None

    # ---- DETECT BAT ----
    bat_res = bat_model(frame, conf=0.25, verbose=False)
    for b in bat_res[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cx,cy = (x1+x2)//2,(y1+y2)//2
        bat_centers.append((cx,cy))
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,255),2)

    # ---- DETECT BALL ----
    ball_res = ball_model(frame, conf=0.25, verbose=False)
    if len(ball_res[0].boxes):
        if prev_ball:
            best = min(ball_res[0].boxes, key=lambda b: euclidean(
                ((b.xyxy[0][0]+b.xyxy[0][2])/2, (b.xyxy[0][1]+b.xyxy[0][3])/2),
                prev_ball))
        else:
            best = ball_res[0].boxes[0]
        x1,y1,x2,y2 = map(int,best.xyxy[0])
        cx,cy = (x1+x2)//2,(y1+y2)//2
        px,py = kalman.update(cx,cy)
        ball_center = (px,py)
    else:
        px,py = kalman.update()
        ball_center = (px,py)

    ball_history.append(ball_center)
    cv2.circle(vis, ball_center, 5, (0,255,0), -1)

    # ---- IMPACT DETECTION ----
    impact = False
    clip_needed = False
    if bat_centers and len(ball_history)>=3:
        closest_bat = min(bat_centers, key=lambda b: euclidean(b, ball_center))
        dist = euclidean(closest_bat, ball_center)

        if dist < IMPACT_DIST:
            clip_needed = True
            score = 0.0

            # Velocity & acceleration
            v1 = (ball_history[-2][0]-ball_history[-3][0], ball_history[-2][1]-ball_history[-3][1])
            v2 = (ball_history[-1][0]-ball_history[-2][0], ball_history[-1][1]-ball_history[-2][1])
            vel1 = np.linalg.norm(v1)
            vel2 = np.linalg.norm(v2)
            accel = abs(vel2-vel1)
            if vel2>VELOCITY_THRESHOLD: score+=0.3
            if accel>ACCEL_THRESHOLD: score+=0.2

            # Optical flow
            x0,y0 = max(0, closest_bat[0]-FLOW_PATCH_SIZE), max(0, closest_bat[1]-FLOW_PATCH_SIZE)
            x1_,y1_ = min(w, closest_bat[0]+FLOW_PATCH_SIZE), min(h, closest_bat[1]+FLOW_PATCH_SIZE)
            prev_patch = prev_frame[y0:y1_, x0:x1_]
            curr_patch = gray[y0:y1_, x0:x1_]
            flow = cv2.calcOpticalFlowFarneback(prev_patch, curr_patch, None, 0.5,3,15,3,5,1.2,0)
            mag,_ = cv2.cartToPolar(flow[...,0], flow[...,1])
            if np.mean(mag)>1.5: score+=0.2

            # Pixel difference
            diff = cv2.absdiff(prev_patch, curr_patch)
            if np.mean(diff)>PIXEL_DIFF_THRESHOLD: score+=0.1

            # Distance bonus
            if dist<IMPACT_DIST: score+=0.2

            if score>=IMPACT_SCORE_THRESHOLD and (i-last_impact>IMPACT_COOLDOWN):
                impact = True
                last_impact = i

    # ---- R3D TEMPORAL MODEL ----
    if clip_needed:
        s = max(0,i-CLIP_LEN//2)
        e = min(len(frames), s+CLIP_LEN)
        clip = preprocess_clip(frames[s:e])
        with torch.no_grad():
            out_r3d = model(clip).squeeze()
            prob = torch.sigmoid(out_r3d).item()
        if prob>0.5:
            impact = True
            last_impact = i

    prev_frame = gray.copy()
    prev_ball = ball_center

    # ---- DRAW IMPACT ----
    if impact:
        cv2.circle(vis, closest_bat, 40, (0,0,255),3)
        cv2.putText(vis, "IMPACT", (w-250,100), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0,0,255), 5, cv2.LINE_AA)
        impact_frames_detected.append(i)

    out.write(vis)

out.release()
print("✅ Impact detection complete! Video saved to:", OUTPUT_PATH)
print("Total impacts detected:", len(impact_frames_detected))
