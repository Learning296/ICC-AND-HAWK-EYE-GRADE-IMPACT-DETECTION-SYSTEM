import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import math
import fielder_biomechanics as biomechanics

# --- Model Loading ---
ball_model = YOLO('runs/detect/train/weights/besst.pt')
pose_model = YOLO('yolov8n-pose.pt')

# --- Constants ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
PERSON_AVG_HEIGHT_M = 1.75
PICKUP_DISTANCE_THRESHOLD_M = 0.5 # in meters
THROW_EVENT_COOLDOWN_FRAMES = 30

# --- Global State ---
def reset_analysis_state():
    """Resets all global variables for a new video analysis."""
    global pixels_per_meter, ball_history, fielder_data, events, frame_count
    pixels_per_meter = None
    ball_history = deque(maxlen=30)
    # fielder_data will store everything about each tracked fielder
    fielder_data = defaultdict(lambda: {
        "positions": deque(maxlen=30),
        "keypoints_history": deque(maxlen=30),
        "last_keypoints": None,
        "is_holding_ball": False,
        "last_pickup_frame": 0,
        "last_throw_frame": 0,
        "last_throw_speed": 0,
        "pickup_release_time": 0,
        "lateral_movement_m": 0,
        "readiness_score": 0,
        "response_time_ms": 0,
    })
    events = []
    frame_count = 0

reset_analysis_state()

def setup_scaling_factor(frame, results_pose):
    """Calibrates pixels_per_meter based on the height of a detected person."""
    global pixels_per_meter
    if results_pose and results_pose[0].keypoints:
        for kps_obj in results_pose[0].keypoints:
            kps = kps_obj.data[0]
            # Use points 0 (nose) and 16 (right ankle) for height estimation
            if kps[0, 2] > 0.5 and kps[16, 2] > 0.5:
                head_y = kps[0, 1]
                ankle_y = kps[16, 1]
                pixel_height = abs(head_y - ankle_y)
                if pixel_height > 100: # A reasonable minimum height in pixels
                    pixels_per_meter = pixel_height / PERSON_AVG_HEIGHT_M
                    print(f"--- Scale Established: {pixels_per_meter:.2f} px/m based on person height. ---")
                    return True
    print("--- Could not find a suitable person for scaling. Using default. ---")
    pixels_per_meter = 50.0 # Fallback default
    return False

def draw_sidebar(frame, w, h):
    """Draws the sidebar with live and event-based stats for each fielder."""
    sidebar_width = 400
    sidebar = np.zeros((h, sidebar_width, 3), dtype=np.uint8)
    
    y_pos = 40
    for fielder_id, data in fielder_data.items():
        if not data["positions"]: continue

        title = f"FIELDER {fielder_id}"
        cv2.putText(sidebar, title, (20, y_pos), FONT, 0.8, (0, 255, 255), 2)
        cv2.line(sidebar, (20, y_pos + 10), (sidebar_width - 20, y_pos + 10), (0, 255, 255), 1)
        y_pos += 35

        def draw_metric(label, value, y, color=(255, 255, 255)):
            cv2.putText(sidebar, f"{label}: {value}", (25, y), FONT, 0.7, color, 1)

        # Live Metrics
        draw_metric("Readiness Score", f"{data['readiness_score']:.1f} %", y_pos)
        y_pos += 30
        draw_metric("Lateral Agility", f"{data['lateral_movement_m']:.2f} m", y_pos)
        y_pos += 45

        # Last Event Metrics
        if data['last_throw_speed'] > 0:
            draw_metric("Last Throw Speed", f"{data['last_throw_speed']:.1f} km/h", y_pos, (50, 205, 255))
            y_pos += 30
        if data['pickup_release_time'] > 0:
            draw_metric("Pickup-to-Throw", f"{data['pickup_release_time']:.2f} s", y_pos, (50, 205, 255))
            y_pos += 30
        if data['response_time_ms'] > 0:
            draw_metric("Response Time", f"{data['response_time_ms']:.0f} ms", y_pos, (50, 205, 255))
            y_pos += 30
        
        y_pos += 40 # Spacer for next fielder

    return np.hstack((frame, sidebar))

def draw_skeleton(frame, keypoints, conf_threshold=0.5):
    """Draws the pose skeleton on the frame."""
    skeleton_connections = [(5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
    points = {i: (int(keypoints[i, 0]), int(keypoints[i, 1])) for i in range(len(keypoints)) if keypoints[i, 2] > conf_threshold}

    for p1_idx, p2_idx in skeleton_connections:
        if p1_idx in points and p2_idx in points:
            cv2.line(frame, points[p1_idx], points[p2_idx], (255, 255, 0), 2)
    for idx, point in points.items():
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
    return points


def analyze_video(input_path, output_path):
    """Main video processing loop."""
    global frame_count, pixels_per_meter
    reset_analysis_state()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video: {input_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w + 400, h))

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_count += 1
        annotated_frame = frame.copy()

        # Calibrate scaling factor on first few frames
        if pixels_per_meter is None:
            results_pose_calib = pose_model(frame, verbose=False)
            setup_scaling_factor(frame, results_pose_calib)

        # --- Object & Pose Detection with Tracking ---
        results_pose = pose_model(frame, conf=0.4, tracker='bytetrack.yaml', verbose=False)
        results_ball = ball_model(frame, conf=0.2, verbose=False)
        
        ball_centers = [((int(box.xyxy[0][0] + box.xyxy[0][2]) // 2), (int(box.xyxy[0][1] + box.xyxy[0][3]) // 2)) for box in results_ball[0].boxes]
        if ball_centers:
            ball_history.append((frame_count, ball_centers[0]))
            for box in results_ball[0].boxes: cv2.rectangle(annotated_frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 255), 2)

        if results_pose and results_pose[0].boxes and results_pose[0].keypoints:
            boxes = results_pose[0].boxes.data.cpu().numpy()
            keypoints_list = results_pose[0].keypoints.data.cpu().numpy()

            for i, box in enumerate(boxes):
                tracker_id = int(box[4])
                
                # Link tracker to keypoints
                kps = keypoints_list[i]
                current_points = draw_skeleton(annotated_frame, kps)
                
                # --- Update Fielder History & Live Metrics ---
                data = fielder_data[tracker_id]
                center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                data["positions"].append((frame_count, (center_x, center_y)))
                data["last_keypoints"] = current_points
                data["keypoints_history"].append((frame_count, current_points))
                
                # Readiness Score
                if 13 in current_points and 14 in current_points and 15 in current_points and 16 in current_points:
                    knee_angle_l = biomechanics.calculate_angle(current_points.get(11), current_points.get(13), current_points.get(15))
                    knee_angle_r = biomechanics.calculate_angle(current_points.get(12), current_points.get(14), current_points.get(16))
                    cog = biomechanics.calculate_center_of_gravity(current_points)
                    if cog:
                        data["readiness_score"] = biomechanics.get_readiness_score(min(knee_angle_l, knee_angle_r), cog[1], h)
                
                # Lateral Agility
                data["lateral_movement_m"] = biomechanics.calculate_lateral_agility(list(data["positions"]), pixels_per_meter)
                
                # --- Event Detection ---
                if ball_centers and current_points:
                    hand_pos = current_points.get(10) or current_points.get(9) # Right or left wrist
                    if hand_pos:
                        dist_to_ball = biomechanics.calculate_distance(hand_pos, ball_centers[0])
                        
                        # PICKUP / CATCH detection
                        if not data["is_holding_ball"] and dist_to_ball < (PICKUP_DISTANCE_THRESHOLD_M * pixels_per_meter):
                            data["is_holding_ball"] = True
                            data["last_pickup_frame"] = frame_count
                            data["response_time_ms"] = (frame_count - data["positions"][-2][0]) / fps * 1000 if len(data["positions"]) > 1 else 0
                            events.append(f"Frame {frame_count}: Fielder {tracker_id} picked up the ball.")
                        
                        # THROW detection
                        elif data["is_holding_ball"] and (frame_count - data["last_throw_frame"]) > THROW_EVENT_COOLDOWN_FRAMES:
                            # If ball is suddenly far from hand after being close
                            if dist_to_ball > (PICKUP_DISTANCE_THRESHOLD_M * pixels_per_meter) * 1.5:
                                data["is_holding_ball"] = False
                                data["last_throw_frame"] = frame_count
                                data["last_throw_speed"] = biomechanics.calculate_speed_kmh(list(ball_history), fps, pixels_per_meter)
                                if data["last_pickup_frame"] > 0:
                                    data["pickup_release_time"] = (frame_count - data["last_pickup_frame"]) / fps
                                events.append(f"Frame {frame_count}: Fielder {tracker_id} threw ball at {data['last_throw_speed']:.1f} km/h.")


        # --- Display ---
        display_frame = draw_sidebar(annotated_frame, w, h)
        out.write(display_frame)
        if __name__ == "__main__":
            cv2.imshow("Fielder Analysis", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    if __name__ == "__main__": cv2.destroyAllWindows()

    print(f"\nProcessing complete. Output saved to {output_path}")
    print("\n--- Detected Events ---")
    for event in events: print(event)
    
    # Final summary can be built here from fielder_data
    return {}

def main():
    """Standalone script entry point."""
    input_path = 'fielder_video.mp4' 
    output_path = 'fielder_analysis_output_1.mp4'
    analyze_video(input_path, output_path)

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, FileNotFoundError) as e:
        print(f"\nProcess stopped or file not found: {e}")
