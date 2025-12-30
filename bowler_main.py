
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import bowler_biomechanics as biomechanics  # Import the new module

# --- Model Loading ---
# Load YOLO models for ball, stump, and pose detection
ball_model = YOLO('runs/detect/train/weights/besst.pt')  # Path to ball detection model
stump_model = YOLO('runs/detect/train_stumps/weights/stumpsweight.pt')  # Path to stump detection model
pose_model = YOLO('yolov8n-pose.pt')  # Pre-trained YOLOv8 pose model

# --- Dynamically Find Class Indices ---
# Find class indices for 'Stumps' and 'Ball' in their respective models
stump_class_index = next((k for k, v in stump_model.names.items() if v.lower() in ['stumps', 'stump']), 0)
ball_class_index = next((k for k, v in ball_model.names.items() if v.lower() in ['sports ball', 'ball', 'cricket_ball', 'cricket-ball']), -1)

print(f"Stump class index: {stump_class_index}, Ball class index: {ball_class_index}")

# --- Constants ---
STUMP_HEIGHT_METERS = 0.711  # Standard cricket stump height in meters
RELEASE_DISPLAY_FRAMES = 30  # Frames to display release information
RELEASE_COOLDOWN_FRAMES = 25  # Minimum frames between consecutive releases
FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font for text overlay
MIN_SPEED_THRESHOLD = 50  # Minimum speed in km/h to consider a release valid
MAX_SPEED_THRESHOLD = 180  # Maximum plausible speed in km/h
FAST_BOWL_THRESHOLD = 130 # Speed in km/h to classify a "Fast" delivery

# --- Global State (Reset for each analysis) ---
def reset_analysis_state():
    """Resets all global variables to their initial state."""
    global pixels_per_meter, bowling_hand_history, ball_history, runup_positions
    global last_release_frame, last_release_speed, release_count, last_release_location
    global processing_stats, biomechanics_stats, bowler_height
    
    pixels_per_meter = None
    bowling_hand_history = deque(maxlen=15) # Increased size for better peak speed calculation
    ball_history = deque(maxlen=10)
    runup_positions = deque(maxlen=30) # Store positions for run-up velocity calculation
    
    last_release_frame = -RELEASE_COOLDOWN_FRAMES - 1
    last_release_speed = 0
    release_count = 0
    last_release_location = None
    bowler_height = 1.8  # Default height in meters, can be updated
    
    processing_stats = {
        "frame_count": 0,
        "releases": [],
        "run_up_velocities": [],
        "stride_lengths": [],
        "release_heights": []
    }
    
    biomechanics_stats = {
        "last_known_angles": {},  # Store latest angles
        "trunk_stability": [],    # Track trunk stability over time
        "arm_speed_rpm": [],     # Track arm rotational speed
        "knee_flexion": [],      # Track knee angles
        "stride_percentage": []   # Track stride length as percentage of height
    }

# Initialize state
reset_analysis_state()


# --- Helper Functions ---

def get_bowling_speed_category(speed_kmh):
    """Categorizes the speed of a bowling delivery."""
    if speed_kmh >= 140:
        return "Express Pace"
    elif speed_kmh >= 130:
        return "Fast"
    elif speed_kmh >= 120:
        return "Fast-Medium"
    elif speed_kmh >= 100:
        return "Medium"
    elif speed_kmh >= 80:
        return "Spin/Slow"
    else:
    return "N/A"

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_peak_speed(history, fps):
    """
    Calculates the PEAK speed from the last few frames to find the true power of a swing.
    This is more robust than averaging or using only two points.
    """
    if len(history) < 2 or pixels_per_meter is None:
        return 0
    
    max_speed = 0
    # Iterate through consecutive pairs of points in the history
    for i in range(len(history) - 1):
        (frame1, pos1), (frame2, pos2) = history[i], history[i+1]
        
        frame_diff = frame2 - frame1
        if frame_diff <= 0:
            continue
            
        pixel_dist = calculate_distance(pos1, pos2)
        time_interval = frame_diff / fps
        
        if time_interval <= 0:
            continue

        pixel_speed_per_sec = pixel_dist / time_interval
        meter_speed_per_sec = pixel_speed_per_sec / pixels_per_meter
        kmh = meter_speed_per_sec * 3.6
        
        if kmh > max_speed:
            max_speed = kmh
            
    return max_speed

def detect_stumps(frame):
    """Detect stumps in the frame and return the height of the best detection."""
    results = stump_model(frame, conf=0.25, verbose=False)
    if results and results[0].boxes:
        stump_boxes = [b for b in results[0].boxes if int(b.cls) == stump_class_index]
        if stump_boxes:
            best_stump = max(stump_boxes, key=lambda x: x.conf)
            return best_stump.xywh[0][3].item()  # Height in pixels
    return None

def setup_scaling_factor(cap):
    """Set up the scaling factor by detecting stumps in the first 150 frames."""
    print("--- Searching for stumps to set scale... ---")
    for i in range(150):
        success, frame = cap.read()
        if not success:
            break
        stump_height = detect_stumps(frame)
        if stump_height and stump_height > 20:  # Ensure detection is reasonable
            ppm = stump_height / STUMP_HEIGHT_METERS
            print(f"--- Scale Established (frame {i}): {ppm:.2f} px/m ---")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
            return ppm
    print("--- Could not find stumps. Using default scale. ---")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return 100.0  # Default scale if stumps not found

def draw_scoreboard_sidebar(frame, release_active, current_speed, last_release_speed, release_count):
    """Draw a clean, organized sidebar with all biomechanical data."""
    h, w, _ = frame.shape
    sidebar_width = 350  # Increased width for more space
    
    # Create a black sidebar on the right
    sidebar = np.zeros((h, sidebar_width, 3), dtype=np.uint8)
    frame_with_sidebar = np.hstack((frame, sidebar))
    
    # Helper for drawing text and titles
    def draw_title(text, y):
        cv2.putText(frame_with_sidebar, text, (w + 20, y), FONT, 0.8, (0, 255, 255), 2)
        cv2.line(frame_with_sidebar, (w + 20, y + 10), (w + sidebar_width - 20, y + 10), (0, 255, 255), 1)

    def draw_metric(label, value, y, color=(255, 255, 255)):
        cv2.putText(frame_with_sidebar, f"{label}: {value}", (w + 25, y), FONT, 0.7, color, 1)

    # --- Live Metrics ---
    y_pos = 40
    draw_title("LIVE ANALYSIS", y_pos)
    
    y_pos += 40
    status_text = "RELEASE!" if release_active else "---"
    status_color = (0, 0, 255) if release_active else (255, 255, 255)
    cv2.putText(frame_with_sidebar, f"Status: {status_text}", (w + 25, y_pos), FONT, 0.7, status_color, 2 if release_active else 1)
    
    y_pos += 35
    draw_metric("Arm Speed", f"{current_speed:.1f} km/h", y_pos, color=(255, 255, 0))
    
    y_pos += 35
    if biomechanics_stats["arm_speed_rpm"]:
        rpm = biomechanics_stats["arm_speed_rpm"][-1]
        draw_metric("Arm RPM", f"{rpm:.1f}", y_pos, color=(255, 255, 0))
    
    # --- Last Delivery Stats ---
    if release_count > 0:
        y_pos = 220
        draw_title(f"LAST DELIVERY (Total: {release_count})", y_pos)
        
        y_pos += 40
        draw_metric("Release Speed", f"{last_release_speed:.1f} km/h", y_pos, color=(50, 205, 255))
        
        last_release_data = processing_stats["releases"][-1]["biomechanics"]
        
        y_pos += 35
        if "stride_percentage" in last_release_data:
            draw_metric("Stride", f"{last_release_data['stride_percentage']:.1f}% of Height", y_pos)
            
        y_pos += 35
        if "release_height" in last_release_data:
            draw_metric("Release Height", f"{last_release_data['release_height']:.2f} m", y_pos)
        
        y_pos += 35
        if "trunk_stability" in last_release_data:
            draw_metric("Trunk Stability", f"{last_release_data['trunk_stability']:.1f}°", y_pos)
            
        y_pos += 35
        if "angles" in last_release_data and "front_knee" in last_release_data["angles"]:
            draw_metric("Knee Flexion", f"{last_release_data['angles']['front_knee']:.1f}°", y_pos)
            
        y_pos += 35
        if "angles" in last_release_data and "bowling_arm" in last_release_data["angles"]:
            draw_metric("Arm Angle", f"{last_release_data['angles']['bowling_arm']:.1f}°", y_pos)
                
    # --- Optimal Ranges Guide ---
    y_pos = h - 160
    draw_title("OPTIMAL RANGES", y_pos)
    
    def draw_guide(text, y):
         cv2.putText(frame_with_sidebar, text, (w + 25, y), FONT, 0.6, (180, 180, 180), 1)
    
    y_pos += 30
    draw_guide("Stride: 81-90% of Height", y_pos)
    y_pos += 25
    draw_guide("Trunk Stability: < 10°", y_pos)
    y_pos += 25
    draw_guide("Knee Flexion: 15-20°", y_pos)
    y_pos += 25
    draw_guide("Arm Angle: > 150°", y_pos)

    return frame_with_sidebar

def draw_skeleton_and_angles(frame, keypoints, conf_threshold=0.5):
    """Draws the batsman's skeleton and key joint angles on the frame."""
    # Define connections between keypoints to form a skeleton
    skeleton_connections = [
        # Torso
        (5, 6), (5, 11), (6, 12), (11, 12),
        # Left Arm
        (5, 7), (7, 9),
        # Right Arm
        (6, 8), (8, 10),
        # Left Leg
        (11, 13), (13, 15),
        # Right Leg
        (12, 14), (14, 16)
    ]

    kps = keypoints.data[0] # Get tensor for the first person
    points = {}
    for i in range(len(kps)):
        if kps[i, 2] > conf_threshold: # Check confidence
            points[i] = (int(kps[i, 0]), int(kps[i, 1]))

    # Draw skeleton lines
    for p1_idx, p2_idx in skeleton_connections:
        if p1_idx in points and p2_idx in points:
            cv2.line(frame, points[p1_idx], points[p2_idx], (255, 255, 0), 2)

    # Draw keypoints
    for idx, point in points.items():
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    # --- Calculate and Display Angles ---
    global biomechanics_stats
    
    # Left Elbow Angle (left_shoulder, left_elbow, left_wrist)
    if 5 in points and 7 in points and 9 in points:
        left_elbow_angle = biomechanics.calculate_angle(points[5], points[7], points[9])
        cv2.putText(frame, f"{left_elbow_angle:.1f}", (points[7][0] + 10, points[7][1]), FONT, 0.6, (0, 255, 0), 2)
        biomechanics_stats["last_known_angles"]["left_elbow"] = left_elbow_angle

    # Right Elbow Angle (right_shoulder, right_elbow, right_wrist)
    if 6 in points and 8 in points and 10 in points:
        right_elbow_angle = biomechanics.calculate_angle(points[6], points[8], points[10])
        cv2.putText(frame, f"{right_elbow_angle:.1f}", (points[8][0] + 10, points[8][1]), FONT, 0.6, (0, 255, 0), 2)
        biomechanics_stats["last_known_angles"]["right_elbow"] = right_elbow_angle

    # Front Knee Angle (hip, knee, ankle) - Assuming right leg is the front leg for a right-arm bowler
    if 12 in points and 14 in points and 16 in points:
        front_knee_angle = biomechanics.calculate_angle(points[12], points[14], points[16])
        cv2.putText(frame, f"Knee: {front_knee_angle:.1f}", (points[14][0] - 80, points[14][1]), FONT, 0.6, (0, 255, 255), 2)
        biomechanics_stats["last_known_angles"]["front_knee"] = front_knee_angle

    # Shoulder-Elbow-Wrist Angle for Bowling Arm (assuming right arm)
    if 6 in points and 8 in points and 10 in points:
        bowling_arm_angle = biomechanics.calculate_angle(points[6], points[8], points[10])
        cv2.putText(frame, f"Arm: {bowling_arm_angle:.1f}", (points[8][0] + 10, points[8][1] - 20), FONT, 0.6, (0, 255, 0), 2)
        biomechanics_stats["last_known_angles"]["bowling_arm"] = bowling_arm_angle

def detect_release(bowling_hand_center, ball_centers, fps, hand_history, threshold, keypoints=None):
    """Detect ball release and calculate speed based on hand movement."""
    global last_release_frame, last_release_speed, release_count, last_release_location, processing_stats
    if not bowling_hand_center or not ball_centers:
        return False, None
        
    frame_count = processing_stats['frame_count']
    if (frame_count - last_release_frame) < RELEASE_COOLDOWN_FRAMES:
        return False, None
        
    min_distance = float('inf')
    release_location = None
        for ball_center in ball_centers:
        distance = calculate_distance(bowling_hand_center, ball_center)
            if distance < min_distance:
                min_distance = distance
            release_location = bowling_hand_center # Release point is the hand's location
            
    # Check if the ball has just left the hand
    if min_distance > threshold:
        arm_speed = calculate_peak_speed(hand_history, fps)
        
        if MIN_SPEED_THRESHOLD < arm_speed < MAX_SPEED_THRESHOLD:
            last_release_speed = arm_speed
            last_release_frame = frame_count
            release_count += 1
            last_release_location = release_location
            
            # Calculate additional biomechanical measurements
            biomechanics_data = {}
            
            # Calculate arm rotational speed (RPM)
            estimated_arm_length = bowler_height * biomechanics.AVERAGE_ARM_LENGTH
            arm_rpm = biomechanics.calculate_arm_rpm(arm_speed, estimated_arm_length)
            biomechanics_stats["arm_speed_rpm"].append(arm_rpm)
            biomechanics_data["arm_speed_rpm"] = arm_rpm
            
            if keypoints is not None:
                # Get keypoint positions
                kps = keypoints.data[0]
                points = {i: (int(kps[i, 0]), int(kps[i, 1])) 
                         for i in range(len(kps)) if kps[i, 2] > 0.5}
                
                # Calculate stride length if we have both feet
                if 15 in points and 16 in points:  # Assuming these are ankle keypoints
                    stride_length = biomechanics.calculate_stride_length(
                        points[15], points[16], pixels_per_meter
                    )
                    stride_percentage = biomechanics.calculate_stride_percentage(
                        stride_length, bowler_height
                    )
                    processing_stats["stride_lengths"].append(stride_length)
                    biomechanics_stats["stride_percentage"].append(stride_percentage)
                    biomechanics_data["stride_length"] = stride_length
                    biomechanics_data["stride_percentage"] = stride_percentage
                
                # Calculate release height
                if release_location and 16 in points:  # Use right ankle as ground reference
                    release_height = biomechanics.calculate_release_height(
                        release_location, points[16], pixels_per_meter
                    )
                    processing_stats["release_heights"].append(release_height)
                    biomechanics_data["release_height"] = release_height
                
                # Calculate trunk stability
                if 11 in points and 12 in points:  # Hip points
                    hip_center = ((points[11][0] + points[12][0])/2,
                                (points[11][1] + points[12][1])/2)
                if 5 in points and 6 in points:  # Shoulder points
                    shoulder_center = ((points[5][0] + points[6][0])/2,
                                     (points[5][1] + points[6][1])/2)
                    # Create vertical reference point above hip
                    vertical_point = (hip_center[0], hip_center[1] - 100)
                    trunk_angle = biomechanics.calculate_trunk_stability(
                        hip_center, shoulder_center, vertical_point
                    )
                    biomechanics_stats["trunk_stability"].append(trunk_angle)
                    biomechanics_data["trunk_stability"] = trunk_angle
            
            # Calculate run-up velocity if we have enough positions
            if len(runup_positions) >= 2:
                runup_velocity = biomechanics.calculate_runup_velocity(
                    list(runup_positions), fps, pixels_per_meter
                )
                processing_stats["run_up_velocities"].append(runup_velocity)
                biomechanics_data["runup_velocity"] = runup_velocity
            
            # Store detailed release data
            release_data = {
                "frame": frame_count,
                "speed_kmh": round(arm_speed, 2),
                "category": biomechanics.get_bowling_speed_category(arm_speed),
                "location": release_location,
                "biomechanics": {
                    **biomechanics_data,
                    "angles": biomechanics_stats.get("last_known_angles", {}).copy()
                }
            }
            processing_stats["releases"].append(release_data)

            print(f"\n>>> RELEASE! Speed: {last_release_speed:.1f} km/h at frame {frame_count}")
            if "stride_percentage" in biomechanics_data:
                print(f"Stride: {biomechanics_data['stride_percentage']:.1f}% of height")
            if "release_height" in biomechanics_data:
                print(f"Release Height: {biomechanics_data['release_height']:.2f}m")
            if "trunk_stability" in biomechanics_data:
                print(f"Trunk Angle: {biomechanics_data['trunk_stability']:.1f}°")
            
            # Clear histories to prepare for the next delivery
            hand_history.clear()
            return True, min_distance
            
    return False, min_distance

def analyze_video(input_path, output_path):
    """
    Process the video, save annotated video, and return analysis statistics.
    """
    # Reset state for a new analysis run
    reset_analysis_state()
    
    global pixels_per_meter, last_release_frame, last_release_speed, release_count, last_release_location, processing_stats
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"ERROR: Could not open video file {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sidebar_width = 350
    print(f"Video Info: {w}x{h} @ {fps:.2f} FPS. Output will be {w + sidebar_width}x{h}")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w + sidebar_width, h))

    pixels_per_meter = setup_scaling_factor(cap)
    release_distance_threshold = 0.8 * pixels_per_meter # Threshold for hand-ball distance

    print("--- Starting video processing ---")
    processing_stats['frame_count'] = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        processing_stats['frame_count'] += 1
        annotated_frame = frame.copy()
        ball_centers, bowling_hand_center = [], None
          
        # Ball Detection
        results_ball = ball_model(frame, conf=0.15, verbose=False)
        if results_ball and results_ball[0].boxes:
            detections = [b for b in results_ball[0].boxes if ball_class_index == -1 or int(b.cls) == ball_class_index]
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                ball_centers.append(ball_center)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Ball ({box.conf.item():.2f})", (x1, y1 - 10), FONT, 0.5, (0, 255, 255), 2)
          
        # Pose Detection (to find the bowler and track the bowling hand)
        results_pose = pose_model(frame, verbose=False)
        best_bowler_idx = -1
        if results_pose and results_pose[0].keypoints is not None:
            # For simplicity, assume the most confident person is the bowler
            if results_pose[0].boxes:
                best_bowler_idx = np.argmax([p.conf for p in results_pose[0].boxes])
            
            if best_bowler_idx != -1:
                bowler_kps_obj = results_pose[0].keypoints[best_bowler_idx]
                bowler_kps_tensor = bowler_kps_obj.data[0]
                
                # Track bowler's position for run-up velocity calculation
                if len(bowler_kps_tensor) > 12 and bowler_kps_tensor[11:13, 2].mean() > 0.5:
                    hip_center = (
                        int((bowler_kps_tensor[11, 0] + bowler_kps_tensor[12, 0]) / 2),
                        int((bowler_kps_tensor[11, 1] + bowler_kps_tensor[12, 1]) / 2)
                    )
                    runup_positions.append((processing_stats['frame_count'], hip_center))
                
                # Assume right-arm bowler (right wrist is keypoint 10)
                if len(bowler_kps_tensor) > 10 and bowler_kps_tensor[10, 2] > 0.5:
                    bowling_hand_center = (int(bowler_kps_tensor[10, 0]), int(bowler_kps_tensor[10, 1]))
                    bowling_hand_history.append((processing_stats['frame_count'], bowling_hand_center))

                draw_skeleton_and_angles(annotated_frame, bowler_kps_obj)
        
        # Release Detection with keypoints for biomechanics
        release_detected, min_dist = detect_release(
            bowling_hand_center, 
            ball_centers, 
            fps, 
            bowling_hand_history, 
            release_distance_threshold,
            keypoints=results_pose[0].keypoints[best_bowler_idx] if best_bowler_idx != -1 and results_pose else None
        )
        
        if release_detected and last_release_location:
            cv2.circle(annotated_frame, last_release_location, 40, (0, 0, 255), 3)
        
        current_arm_speed = calculate_peak_speed(bowling_hand_history, fps)
        
        print(f"\rFrame: {processing_stats['frame_count']}, Live Arm Speed: {current_arm_speed:.1f} km/h", end="")

        release_active = (processing_stats['frame_count'] - last_release_frame) < RELEASE_DISPLAY_FRAMES
        display_frame = draw_scoreboard_sidebar(
            annotated_frame, release_active, current_arm_speed, last_release_speed, release_count
        )

        out.write(display_frame)
        if __name__ == "__main__":
            cv2.imshow("Cricket Analysis", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    out.release()
    if __name__ == "__main__":
        cv2.destroyAllWindows()
    print(f"\nProcessing complete. Output saved to {output_path}")

    # --- Final Statistics ---
    final_stats = {
        "total_frames": processing_stats['frame_count'],
        "total_deliveries": release_count,
        "releases": processing_stats["releases"],
        "summary": {}
    }

    if release_count > 0:
        speeds = [rel['speed_kmh'] for rel in processing_stats['releases']]
        
        summary = {
            "average_speed_kmh": round(np.mean(speeds), 2),
            "max_speed_kmh": round(np.max(speeds), 2),
            "fastest_delivery_category": biomechanics.get_bowling_speed_category(np.max(speeds)),
        }

        # Safely extract and average biomechanics data
        biomech_data = [rel.get("biomechanics", {}) for rel in processing_stats["releases"]]
        
        def get_avg_metric(key1, key2=None):
            values = []
            for item in biomech_data:
                if key2:
                    val = item.get(key1, {}).get(key2)
                else:
                    val = item.get(key1)
                if val is not None:
                    values.append(val)
            return round(np.mean(values), 2) if values else 0

        summary["avg_runup_velocity_ms"] = get_avg_metric("runup_velocity")
        summary["avg_stride_percentage"] = get_avg_metric("stride_percentage")
        summary["avg_release_height_m"] = get_avg_metric("release_height")
        summary["avg_trunk_stability_deg"] = get_avg_metric("trunk_stability")
        summary["avg_arm_rpm"] = get_avg_metric("arm_speed_rpm")
        summary["avg_front_knee_angle"] = get_avg_metric("angles", "front_knee")
        summary["avg_bowling_arm_angle"] = get_avg_metric("angles", "bowling_arm")
        
        final_stats["summary"] = summary
    
    return final_stats

def main():
    """Standalone script entry point."""
    input_path = 'new_check_video_1.mp4'
    output_path = 'bowler_analysis_output_final_1.mp4'
    stats = analyze_video(input_path, output_path)
    print("\n--- Bowler Analysis Report ---")
    import json
    print(json.dumps(stats, indent=4))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")


