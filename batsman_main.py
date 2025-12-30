import cv2
import numpy as np
import math
from collections import deque
from ultralytics import YOLO

import batsman_biomechanics as biomechanics
from smart_biomechanics_system import SmartBiomechanicsSystem

from module1_yolo import generate_impact_proposal
from module2_slowfast import SlowFastImpactDetector


# ================== MODEL LOADING ==================
bat_model = YOLO('runs/detect/train/weights/bat_new_2.pt')
ball_model = YOLO('runs/detect/train/weights/besst.pt')
stump_model = YOLO('runs/detect/train_stumps/weights/stumpsweight.pt')
pose_model = YOLO('yolov8n-pose.pt')

stump_class_index = next(
    (k for k, v in stump_model.names.items() if v.lower() in ['stumps', 'stump']), 0
)

ball_class_index = next(
    (k for k, v in ball_model.names.items()
     if v.lower() in ['sports ball', 'ball', 'cricket_ball', 'cricket-ball']), -1
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ================== CONSTANTS ==================
STUMP_HEIGHT_METERS = 0.711
IMPACT_COOLDOWN_FRAMES = 15
MIN_SPEED_THRESHOLD = 15
MAX_SPEED_THRESHOLD = 250


# ================== GLOBAL STATE ==================
def reset_state():
    global bat_history, ball_history, head_history
    global last_impact_frame, impact_count
    global pixels_per_meter, processing_stats
    global smart_system

    bat_history = deque(maxlen=15)
    ball_history = deque(maxlen=10)
    head_history = deque(maxlen=15)

    last_impact_frame = -999
    impact_count = 0
    pixels_per_meter = None

    processing_stats = {
        "frame_count": 0,
        "impacts": []
    }

    smart_system = SmartBiomechanicsSystem()


# ================== UTILS ==================
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def peak_speed(history, fps):
    if len(history) < 2 or pixels_per_meter is None:
        return 0

    speeds = []
    for i in range(len(history)-1):
        (f1, p1), (f2, p2) = history[i], history[i+1]
        dt = (f2 - f1) / fps
        if dt <= 0:
            continue
        px = distance(p1, p2)
        speeds.append((px / pixels_per_meter) / dt * 3.6)

    return max(speeds) if speeds else 0


# ================== IMPACT CORE ==================
def detect_impact(bat_centers, ball_centers, fps, threshold):
    global last_impact_frame, impact_count

    if not bat_centers or not ball_centers:
        return False, None

    if processing_stats["frame_count"] - last_impact_frame < IMPACT_COOLDOWN_FRAMES:
        return False, None

    min_dist, impact_loc = min(
        ((distance(b, c), b) for b in bat_centers for c in ball_centers),
        key=lambda x: x[0],
        default=(9999, None)
    )

    if min_dist < threshold:
        speed = peak_speed(bat_history, fps)
        if MIN_SPEED_THRESHOLD < speed < MAX_SPEED_THRESHOLD:
            impact_count += 1
            last_impact_frame = processing_stats["frame_count"]
            processing_stats["impacts"].append({
                "frame": processing_stats["frame_count"],
                "speed_kmh": round(speed, 2)
            })
            print(f"\nIMPACT CONFIRMED | Speed: {speed:.1f} km/h")
            bat_history.clear()
            return True, impact_loc

    return False, None


# ================== MAIN PIPELINE ==================
def analyze_video(input_path, output_path):
    reset_state()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))

    # Preload all frames for faster access
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    slowfast = SlowFastImpactDetector(threshold=0.7)
    pixels_per_meter = 120
    impact_distance_threshold = 150  # pixels (loose on purpose)

    for idx, frame in enumerate(frames):
        processing_stats["frame_count"] = idx
        annotated = frame.copy()

        # -------- YOLO BAT --------
        bat_res = bat_model(frame, conf=0.25, verbose=False)
        bat_centers = []
        if len(bat_res[0].boxes) > 0:
            for b in bat_res[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2
                bat_centers.append((cx, cy))
                bat_history.append((idx, (cx, cy)))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,165,255), 2)

        # -------- YOLO BALL --------
        ball_res = ball_model(frame, conf=0.2, verbose=False)
        ball_centers = []
        if len(ball_res[0].boxes) > 0:
            for b in ball_res[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2
                ball_centers.append((cx, cy))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,255), 2)

        # -------- Module-1: Proposal --------
        proposal, location = generate_impact_proposal(
            bat_centers, ball_centers, impact_distance_threshold
        )

        if proposal:
            cv2.circle(annotated, location, 25, (0,255,255), 2)

            # -------- Module-2: SlowFast --------
            clip = slowfast.extract_clip(frames, idx, fps)
            confirmed, confidence = slowfast.confirm_impact(clip)

            if confirmed:
                impact, loc = detect_impact(bat_centers, ball_centers, fps, impact_distance_threshold)
                if impact:
                    cv2.circle(annotated, loc, 35, (0,255,0), 3)

        out.write(annotated)

        # Optional for live preview
        # cv2.imshow("Cricket Analysis", annotated)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    out.release()
    cv2.destroyAllWindows()
    print("\nPROCESS COMPLETE")
    print(processing_stats)

if __name__ == "__main__":
    analyze_video(
        "new_video.mp4",  # ← yahi tumhara input video path hai
        "final_output.mp4"       # ← yahi output video save hoga
    )
