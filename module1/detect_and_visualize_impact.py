# detect_and_visualize_impact.py
import cv2
import torch
import numpy as np
from module2_slowfast import x3d_m

# ------------------- SETTINGS -------------------
MODEL_PATH = "slowfast_cricket_final.pth"  # trained model
VIDEO_PATH = "video1.mp4"             # input video
OUTPUT_PATH = "output_impact.mp4"         # output video
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16     # SlowFast input
FRAME_SIZE = 224    # resize for model

# ------------------- LOAD MODEL -------------------
model = x3d_m(pretrained=False)
model.blocks[-1].proj = torch.nn.Linear(2048, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ------------------- VIDEO SETUP -------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frames_buffer = []

# ------------------- HELPER -------------------
def preprocess(frames):
    frames_resized = [cv2.resize(f, (FRAME_SIZE, FRAME_SIZE)) for f in frames]
    frames_array = np.stack(frames_resized).astype(np.float32)/255.0
    frames_tensor = torch.tensor(frames_array).permute(3,0,1,2).unsqueeze(0)  # C,T,H,W with batch
    return frames_tensor

# ------------------- PROCESS VIDEO -------------------
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames_buffer.append(frame)
    
    # Once buffer has enough frames, predict
    if len(frames_buffer) >= NUM_FRAMES:
        input_clip = preprocess(frames_buffer[-NUM_FRAMES:])
        input_clip = input_clip.to(DEVICE)
        with torch.no_grad():
            outputs = model(input_clip)
            pred = torch.argmax(outputs, dim=1).item()
        
        # Draw Impact label on current frame
        label_text = "Impact!" if pred == 1 else "No Impact"
        color = (0,0,255) if pred == 1 else (0,255,0)
        cv2.putText(frames_buffer[-1], label_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Write frame to output video
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ Detection complete! Output saved as {OUTPUT_PATH}")
