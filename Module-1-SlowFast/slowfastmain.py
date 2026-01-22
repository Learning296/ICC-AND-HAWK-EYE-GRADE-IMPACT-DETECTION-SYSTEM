import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ---------------- CONSTANTS ----------------
DIST_THRESHOLD = 20                  # pixels: ball very close to bat
IMPACT_SCORE_THRESHOLD = 0.6         # combined score threshold
IMPACT_COOLDOWN = 5                  # frames to skip after impact
FLOW_PATCH_SIZE = 40                 # size of patch around bat-ball region
VELOCITY_THRESHOLD = 3               # px/frame
ACCEL_THRESHOLD = 2.5                # px/frame^2
PIXEL_DIFF_THRESHOLD = 25             # intensity change threshold

# ---------------- UTILS ----------------
def euclidean(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

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

# ---------------- IMPACT DETECTION PIPELINE ----------------
def run(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    h,w,_ = frames[0].shape
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    # ---------------- MODELS ----------------
    bat_model = YOLO("runs/detect/train/weights/best.pt")
    ball_model = YOLO("runs/detect/train/weights/best.pt")

    kalman = BallKalman()
    last_impact = -999

    prev_ball = None
    prev_velocity = None
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    ball_history = deque(maxlen=5)

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        bat_centers = []
        ball_center = None

        # ---- DETECT BAT ----
        for b in bat_model(frame, conf=0.2, verbose=False)[0].boxes:
            x1,y1,x2,y2 = map(int,b.xyxy[0])
            cx,cy = (x1+x2)//2,(y1+y2)//2
            bat_centers.append((cx,cy))
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,255),2)

        # ---- DETECT BALL ----
        detections = ball_model(frame, conf=0.15, verbose=False)[0].boxes
        if len(detections):
            if prev_ball:
                best = min(detections, key=lambda b: euclidean(
                    ((b.xyxy[0][0]+b.xyxy[0][2])/2, (b.xyxy[0][1]+b.xyxy[0][3])/2),
                    prev_ball))
            else:
                best = detections[0]
            x1,y1,x2,y2 = map(int,best.xyxy[0])
            cx,cy = (x1+x2)//2,(y1+y2)//2
            px,py = kalman.update(cx,cy)
            ball_center = (px,py)
        else:
            px,py = kalman.update()
            ball_center = (px,py)

        ball_history.append(ball_center)
        cv2.circle(vis, ball_center, 5, (0,255,0), -1)

        # ---- IMPACT CHECK ----
        impact = False
        if bat_centers and len(ball_history) >= 3:
            closest_bat = min(bat_centers, key=lambda b: euclidean(b, ball_center))
            dist = euclidean(closest_bat, ball_center)

            if dist < DIST_THRESHOLD:
                score = 0.0

                # --- Velocity & Acceleration Check ---
                if len(ball_history) >=3:
                    v1 = (ball_history[-2][0]-ball_history[-3][0], ball_history[-2][1]-ball_history[-3][1])
                    v2 = (ball_history[-1][0]-ball_history[-2][0], ball_history[-1][1]-ball_history[-2][1])
                    vel1 = np.linalg.norm(v1)
                    vel2 = np.linalg.norm(v2)
                    accel = abs(vel2-vel1)
                    if vel2>VELOCITY_THRESHOLD:
                        score +=0.3
                    if accel>ACCEL_THRESHOLD:
                        score +=0.2

                # --- Optical Flow Check ---
                x0 = max(0, closest_bat[0]-FLOW_PATCH_SIZE)
                y0 = max(0, closest_bat[1]-FLOW_PATCH_SIZE)
                x1_ = min(w, closest_bat[0]+FLOW_PATCH_SIZE)
                y1_ = min(h, closest_bat[1]+FLOW_PATCH_SIZE)
                prev_patch = prev_frame[y0:y1_, x0:x1_]
                curr_patch = gray[y0:y1_, x0:x1_]
                flow = cv2.calcOpticalFlowFarneback(prev_patch, curr_patch, None, 0.5,3,15,3,5,1.2,0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                flow_score = np.mean(mag)
                if flow_score>1.5: # empirical threshold
                    score +=0.2

                # --- Pixel Difference Check ---
                diff = cv2.absdiff(prev_patch, curr_patch)
                if np.mean(diff)>PIXEL_DIFF_THRESHOLD:
                    score +=0.1

                # --- Distance bonus ---
                if dist<DIST_THRESHOLD:
                    score +=0.2

                if score>=IMPACT_SCORE_THRESHOLD and (i-last_impact>IMPACT_COOLDOWN):
                    impact = True
                    last_impact = i

        prev_frame = gray.copy()
        prev_ball = ball_center

        # ---- DRAW IMPACT ----
        if impact:
            # optional small circle on bat
            cv2.circle(vis, closest_bat, 40, (0,0,255),3)

            # large IMPACT label on top-right corner
            cv2.putText(
                vis,
                "IMPACT",
                (w - 250, 100),             # position (adjust as needed)
                cv2.FONT_HERSHEY_DUPLEX,
                2.5,                         # font size
                (0,0,255),                   # red
                5,                           # thickness
                cv2.LINE_AA
            )

        out.write(vis)

    out.release()
    print("DONE! Video saved to", out_path)

# ---------------- RUN ----------------
if __name__=="__main__":
    run("videos/video2.mp4", "output.mp4")
