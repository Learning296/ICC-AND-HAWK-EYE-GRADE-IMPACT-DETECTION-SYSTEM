# =========================================================
# FINAL CRICKET IMPACT DETECTION - MODIFIED
# YOLO + Physics + R3D + SlowFast (PyTorchVideo)
# Improved: peak detection + weighted fusion
# =========================================================

import cv2
import torch
import numpy as np
import csv
from ultralytics import YOLO
from torchvision.models.video import r3d_18
from pytorchvideo.models.hub import slowfast_r50

# ---------------- CONFIG ----------------
VIDEO_PATH = "video1.mp4"
OUTPUT_VIDEO = "output_final.mp4"
OUTPUT_CSV = "output_final.csv"

BAT_MODEL_PATH = "runs/detect/train/weights/bat_new_2.pt"
BALL_MODEL_PATH = "runs/detect/train/weights/besst.pt"

IMPACT_R3D_WEIGHTS = r"C:\Users\Binary Computers\Downloads\module1\models\impact_r3d_augmented.pth"
IMPACT_SLOWFAST_WEIGHTS = r"C:\Users\Binary Computers\Downloads\module1\models\impact_slowfast_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIST_THRESH = 120
ACCEL_THRESH = 12
FINAL_THRESHOLD = 0.6
IMPACT_COOLDOWN = 8   # minimum frame gap between impacts

# ---------------- HELPERS ----------------
def center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

# ---------------- BALL TRACKER ----------------
class BallTracker:
    def __init__(self):
        self.prev_pos = None
        self.prev_speed = None

    def update(self, pos):
        if self.prev_pos is None:
            self.prev_pos = pos
            self.prev_speed = 0.0
            return 0.0

        speed = np.linalg.norm(pos - self.prev_pos)
        accel = abs(speed - self.prev_speed)

        self.prev_pos = pos
        self.prev_speed = speed

        if accel > ACCEL_THRESH:
            return min(accel / 60.0, 1.0)
        return 0.0

# ---------------- R3D MODEL ----------------
class ImpactR3D:
    def __init__(self, weights):
        self.device = DEVICE
        self.model = r3d_18(weights=None)
        self.model.fc = torch.nn.Linear(512, 1)
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.num_frames = 16
        self.size = 112
        self.mean = np.array([0.43216, 0.394666, 0.37645])
        self.std  = np.array([0.22803, 0.22145, 0.216989])

    def preprocess(self, clip):
        if len(clip) < self.num_frames:
            clip += [clip[-1]] * (self.num_frames - len(clip))
        idxs = np.linspace(0, len(clip)-1, self.num_frames).astype(int)
        frames = []
        for i in idxs:
            img = cv2.resize(clip[i], (self.size, self.size))
            img = img[:, :, ::-1] / 255.0
            img = (img - self.mean) / self.std
            frames.append(img)
        x = torch.tensor(np.stack(frames)).permute(3,0,1,2).unsqueeze(0).float()
        return x.to(self.device)

    def predict(self, clip):
        with torch.no_grad():
            prob = torch.sigmoid(self.model(self.preprocess(clip)))[0,0].item()
        return max(0.0, (prob - 0.5) * 2.0)

# ---------------- SLOWFAST MODEL ----------------
class ImpactSlowFast:
    def __init__(self, weights):
        self.device = DEVICE
        self.model = slowfast_r50(pretrained=False)
        self.model.blocks[-1].proj = torch.nn.Linear(
            self.model.blocks[-1].proj.in_features, 1
        )
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.num_fast = 32
        self.alpha = 4
        self.size = 224
        self.mean = np.array([0.45,0.45,0.45])
        self.std  = np.array([0.225,0.225,0.225])

    def preprocess(self, clip):
        if len(clip) < self.num_fast:
            clip += [clip[-1]] * (self.num_fast - len(clip))
        idx = np.linspace(0, len(clip)-1, self.num_fast).astype(int)
        fast = []
        for i in idx:
            img = cv2.resize(clip[i], (self.size, self.size))
            img = img[:, :, ::-1] / 255.0
            img = (img - self.mean) / self.std
            fast.append(img)
        fast = np.stack(fast)
        slow = fast[::self.alpha]
        fast = torch.tensor(fast).permute(3,0,1,2).unsqueeze(0).float().to(self.device)
        slow = torch.tensor(slow).permute(3,0,1,2).unsqueeze(0).float().to(self.device)
        return [slow, fast]

    def predict(self, clip):
        with torch.no_grad():
            prob = torch.sigmoid(self.model(self.preprocess(clip)))[0,0].item()
        return max(0.0, (prob - 0.5) * 2.0)

# ---------------- MAIN ----------------
def run():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frames[0].shape[1], frames[0].shape[0])
    )

    bat_model = YOLO(BAT_MODEL_PATH)
    ball_model = YOLO(BALL_MODEL_PATH)
    r3d = ImpactR3D(IMPACT_R3D_WEIGHTS)
    slowfast = ImpactSlowFast(IMPACT_SLOWFAST_WEIGHTS)
    tracker = BallTracker()

    last_impact_frame = -100
    conf_history = []

    csv_file = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "frame","r3d_conf","slowfast_conf","physics_conf","final_conf","impact"
    ])

    for i, frame in enumerate(frames):
        bat_res = bat_model.predict(frame, verbose=False)[0]
        ball_res = ball_model.predict(frame, verbose=False)[0]

        bat_boxes = bat_res.boxes.xyxy.cpu().numpy() if bat_res.boxes else []
        ball_boxes = ball_res.boxes.xyxy.cpu().numpy() if ball_res.boxes else []

        # Physics confidence
        physics_conf = 0.0
        if len(ball_boxes) > 0:
            physics_conf = tracker.update(center(ball_boxes[0]))

        # Proposal check
        proposal = False
        for b in bat_boxes:
            for bl in ball_boxes:
                if np.linalg.norm(center(b) - center(bl)) < DIST_THRESH:
                    proposal = True

        r3d_conf = 0.0
        slowfast_conf = 0.0
        if proposal:
            r3d_conf = r3d.predict(frames[max(0,i-8):i+8])
            slowfast_conf = slowfast.predict(frames[max(0,i-32):i])

        # Weighted fusion
        final_conf = 0.4*slowfast_conf + 0.4*r3d_conf + 0.2*physics_conf

        # Peak detection for impact
        conf_history.append(final_conf)
        if len(conf_history) > 5:
            conf_history.pop(0)

        impact = False
        if len(conf_history) == 5:
            mid = conf_history[2]
            if (
                mid == max(conf_history) and
                mid >= FINAL_THRESHOLD and
                i - last_impact_frame > IMPACT_COOLDOWN
            ):
                impact = True
                last_impact_frame = i

        if impact:
            cv2.putText(frame, "IMPACT", (50,80),
                        cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),4)

        out.write(frame)

        writer.writerow([
            i,
            round(r3d_conf,3),
            round(slowfast_conf,3),
            round(physics_conf,3),
            round(final_conf,3),
            int(impact)
        ])

        print(f"Frame {i} | R3D={r3d_conf:.2f} | SF={slowfast_conf:.2f} | PHY={physics_conf:.2f} | FINAL={final_conf:.2f} | IMPACT={impact}")

    out.release()
    csv_file.close()
    print("✅ DONE: output_final.mp4 & output_final.csv saved")

# ---------------- RUN ----------------
if __name__ == "__main__":
    run()
