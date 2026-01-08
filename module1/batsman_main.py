import cv2
import torch
import numpy as np
from module2_r3d_impact import ImpactDetector
from ultralytics import YOLO  # YOLOv8 package

# --------- CONFIG ---------
VIDEO_PATH = "videos/video1.mp4"
OUTPUT_PATH = "output.mp4"
IMPACT_WEIGHT_PATH = "weights/impact_model.pth"  # 3D ResNet trained weights
THRESHOLD = 0.3

# YOLO models
BAT_MODEL_PATH = 'runs/detect/train/weights/bat_new_2.pt'
BALL_MODEL_PATH = 'runs/detect/train/weights/besst.pt'

# --------- HELPER FUNCTIONS ---------
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def write_video(frames, output_path, fps):
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

def boxes_iou(box1, box2, threshold=0.05):
    """
    Returns True if IoU of two boxes is above threshold.
    box = [x1, y1, x2, y2]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou > threshold

# --------- MAIN PROCESS ---------
def run(video_path, output_path):
    # Load frames
    frames, fps = read_video(video_path)

    # Load models
    impact_net = ImpactDetector(IMPACT_WEIGHT_PATH, threshold=THRESHOLD)
    bat_model = YOLO(BAT_MODEL_PATH)
    ball_model = YOLO(BALL_MODEL_PATH)

    output_frames = frames.copy()

    for i, frame in enumerate(frames):
        # ---- Impact Detection (3D ResNet) ----
        clip = impact_net.extract_clip(frames, i, fps)
        is_impact, conf = impact_net.predict(clip)

        # ---- Bat Detection ----
        bat_results = bat_model.predict(frame)
        bat_boxes = [list(map(int, det)) for det in bat_results[0].boxes.xyxy]

        for x1, y1, x2, y2 in bat_boxes:
            cv2.rectangle(output_frames[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_frames[i], "Bat", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ---- Ball Detection ----
        ball_results = ball_model.predict(frame)
        ball_boxes = [list(map(int, det)) for det in ball_results[0].boxes.xyxy]

        for x1, y1, x2, y2 in ball_boxes:
            cv2.rectangle(output_frames[i], (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(output_frames[i], "Ball", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # ---- Combined Impact Logic ----
        impact_detected = is_impact  # start with 3D ResNet result

        # Check if any bat box has IoU with any ball box above small threshold
        for b_box in bat_boxes:
            for bl_box in ball_boxes:
                if boxes_iou(b_box, bl_box, threshold=0.05):  # adjust threshold if needed
                    impact_detected = True
                    conf = 1.0  # high confidence for bat-ball collision
                    break

        # ---- Annotate impact ----
        if impact_detected:
            cv2.putText(output_frames[i], f"Impact! ({conf:.2f})",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

    # Save video
    write_video(output_frames, output_path, fps)
    print(f"[INFO] Output saved at {output_path}")

# --------- RUN ---------
if __name__ == "__main__":
    run(VIDEO_PATH, OUTPUT_PATH)
