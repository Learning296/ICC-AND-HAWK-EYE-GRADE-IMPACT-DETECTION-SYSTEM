import math
import numpy as np
from typing import Tuple, List, Dict, Optional

def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """Calculates the angle (in degrees) between three points, where p2 is the vertex."""
    if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in [p1, p2, p3]):
        return 0.0
    a = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
    b = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    c = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
    numerator = a**2 + b**2 - c**2
    denominator = 2 * a * b
    if denominator == 0:
        return 0.0
    cosine_angle = max(-1.0, min(1.0, numerator / denominator))
    return math.degrees(math.acos(cosine_angle))

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_speed_kmh(history: List[Tuple[int, Tuple[float, float]]], fps: float, pixels_per_meter: float) -> float:
    """Calculates peak speed in km/h from a history of positions."""
    if len(history) < 2 or pixels_per_meter is None or fps <= 0:
        return 0.0
    
    max_speed = 0
    for i in range(len(history) - 1):
        frame1, pos1 = history[i]
        frame2, pos2 = history[i+1]
        
        time_diff_s = (frame2 - frame1) / fps
        if time_diff_s <= 0:
            continue
            
        pixel_dist = calculate_distance(pos1, pos2)
        meter_dist = pixel_dist / pixels_per_meter
        speed_ms = meter_dist / time_diff_s
        speed_kmh = speed_ms * 3.6
        
        if speed_kmh > max_speed:
            max_speed = speed_kmh
            
    return max_speed

def calculate_center_of_gravity(keypoints: Dict[int, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Estimates the center of gravity using hip and shoulder keypoints."""
    # Keypoints: 5-LShoulder, 6-RShoulder, 11-LHip, 12-RHip
    if all(k in keypoints for k in [5, 6, 11, 12]):
        l_shoulder = np.array(keypoints[5])
        r_shoulder = np.array(keypoints[6])
        l_hip = np.array(keypoints[11])
        r_hip = np.array(keypoints[12])
        
        shoulder_mid = (l_shoulder + r_shoulder) / 2
        hip_mid = (l_hip + r_hip) / 2
        
        # A simple approximation is the midpoint between the shoulder and hip centers
        cog = (shoulder_mid + hip_mid) / 2
        return tuple(cog)
    return None

def get_readiness_score(knee_angle: float, cog_y: float, frame_height: int) -> float:
    """
    Calculates a readiness score (0-100) based on posture.
    Lower CoG and bent knees result in a higher score.
    """
    if cog_y is None or frame_height <= 0: return 0
    # Normalize CoG position (0 at top, 1 at bottom)
    normalized_cog = cog_y / frame_height
    
    # Ideal knee angle for readiness is around 120-140 degrees (bent)
    knee_score = 0
    if 100 < knee_angle < 160:
        knee_score = 100
    elif knee_angle <= 100:
        knee_score = (knee_angle / 100) * 100
    else: # knee_angle >= 160 (too straight)
        knee_score = ((180 - knee_angle) / 20) * 100

    # Ideal CoG is lower in the body
    cog_score = (normalized_cog) * 100
    
    # Combine scores (70% CoG, 30% Knee Angle)
    readiness = (np.clip(cog_score, 0, 100) * 0.7) + (np.clip(knee_score, 0, 100) * 0.3)
    return min(readiness, 100.0)

def calculate_pickup_to_throw_time(pickup_frame: int, throw_frame: int, fps: float) -> float:
    """Calculates the time in seconds between a pickup and a throw event."""
    if not all([pickup_frame, throw_frame, fps > 0]):
        return 0.0
    return (throw_frame - pickup_frame) / fps

def calculate_lateral_agility(position_history: List[Tuple[int, Tuple[float, float]]], pixels_per_meter: float) -> float:
    """Calculates the total lateral (horizontal) distance covered by a fielder."""
    if len(position_history) < 2 or pixels_per_meter <= 0:
        return 0.0
    
    total_lateral_dist_pixels = 0
    for i in range(len(position_history) - 1):
        _, pos1 = position_history[i]
        _, pos2 = position_history[i+1]
        total_lateral_dist_pixels += abs(pos1[0] - pos2[0])
        
    return total_lateral_dist_pixels / pixels_per_meter
