import math
import numpy as np
from typing import Tuple, List, Dict, Optional

# --- Constants for Batsman Biomechanics ---
AVERAGE_BATSMAN_HEIGHT = 1.75  # in meters, used for normalization
AVERAGE_SHOULDER_WIDTH = 0.45 # in meters, used for stance analysis

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

def calculate_stance_width_ratio(left_ankle: Tuple[float, float], right_ankle: Tuple[float, float], shoulder_width_pixels: float) -> float:
    """
    Calculates the stance width as a ratio of shoulder width.
    A good stance is typically 1.0 to 1.5 times shoulder width.
    """
    if not all([left_ankle, right_ankle, shoulder_width_pixels > 0]):
        return 0.0
    stance_pixel_width = calculate_distance(left_ankle, right_ankle)
    return stance_pixel_width / shoulder_width_pixels

def calculate_backlift_height(bat_top: Tuple[float, float], head_top: Tuple[float, float], pixels_per_meter: float) -> float:
    """
    Calculates the height of the backlift in meters relative to the head.
    """
    if not all([bat_top, head_top, pixels_per_meter > 0]):
        return 0.0
    pixel_height = abs(bat_top[1] - head_top[1])
    return pixel_height / pixels_per_meter

def calculate_head_stability(head_positions: List[Tuple[float, float]]) -> float:
    """
    Calculates head movement during a shot. Returns the standard deviation of head positions.
    Lower is better.
    """
    if len(head_positions) < 2:
        return 0.0
    head_positions_np = np.array(head_positions)
    # Calculate the standard deviation of the x and y coordinates
    std_dev = np.std(head_positions_np, axis=0)
    # Return the magnitude of the standard deviation vector
    return np.linalg.norm(std_dev)

def calculate_shot_power_index(bat_speed_kmh: float) -> float:
    """
    Calculates a simple power index based on bat speed.
    Normalizes the speed to a score out of 100.
    """
    # Assuming a max plausible bat speed of 160 km/h for normalization
    power = (bat_speed_kmh / 160.0) * 100
    return min(power, 100.0) # Cap at 100

def calculate_shoulder_hip_separation(
    left_shoulder: Tuple[float, float], right_shoulder: Tuple[float, float],
    left_hip: Tuple[float, float], right_hip: Tuple[float, float]
) -> float:
    """
    Calculates the separation angle between the line of the shoulders and the hips.
    A larger angle indicates more torque and power.
    """
    if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
        return 0.0
    
    # Calculate the angle of the shoulder and hip lines relative to the horizontal
    shoulder_angle = math.degrees(math.atan2(left_shoulder[1] - right_shoulder[1], left_shoulder[0] - right_shoulder[0]))
    hip_angle = math.degrees(math.atan2(left_hip[1] - right_hip[1], left_hip[0] - right_hip[0]))
    
    separation = abs(shoulder_angle - hip_angle)
    # Normalize to the range [0, 180]
    if separation > 180:
        separation = 360 - separation
    return separation

def get_bat_speed_category(speed_kmh: float) -> str:
    """Categorizes the bat speed."""
    if speed_kmh > 130:
        return "Very Fast"
    elif speed_kmh > 110:
        return "Fast"
    elif speed_kmh > 90:
        return "Medium"
    else:
        return "Slow"
