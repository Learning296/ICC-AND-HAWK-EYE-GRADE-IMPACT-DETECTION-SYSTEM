import cv2
import os

# ---------------- CONFIG ----------------
VIDEO_PATH = r"C:\Users\Binary Computers\Downloads\module1\check_video.mp4"
SAVE_DIR   = r"C:\Users\Binary Computers\Downloads\module1\dataset\no_impact"

START_SEC = 0.6
END_SEC   = 4.10

FIXED_FPS = 30   # hard lock (safe)

# ---------------- SETUP ----------------
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Video open nahi ho rahi")

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

total_frames = len(frames)
print("[INFO] Total frames:", total_frames)

start_frame = int(START_SEC * FIXED_FPS)
end_frame   = int(END_SEC * FIXED_FPS)

start_frame = max(0, start_frame)
end_frame   = min(total_frames, end_frame)

print("[INFO] Extracting frames:", start_frame, "to", end_frame)

clip_frames = frames[start_frame:end_frame]

if len(clip_frames) == 0:
    raise RuntimeError("No frames extracted")

h, w, _ = clip_frames[0].shape

out_path = os.path.join(SAVE_DIR, "shot_0.6_to_4.10.mp4")

writer = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FIXED_FPS,
    (w, h)
)

for f in clip_frames:
    writer.write(f)

writer.release()

print("✅ CLIP SAVED:", out_path)
print("⏱ Duration:", len(clip_frames) / FIXED_FPS, "seconds")
