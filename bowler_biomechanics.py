import math
import numpy as np
from typing import Tuple, List, Dict, Optional

# Constants for biomechanical analysis
OPTIMAL_STRIDE_MIN = 0.81  # Minimum optimal stride length as percentage of height
OPTIMAL_STRIDE_MAX = 0.90  # Maximum optimal stride length as percentage of height
AVERAGE_ARM_LENGTH = 0.73  # Average arm length as percentage of height

def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Calculates the angle (in degrees) between three points, where p2 is the vertex.
    The angle is computed using the law of cosines.
    """
    # Ensure points are valid tuples
    if not (isinstance(p1, (list, tuple)) and len(p1) == 2 and
            isinstance(p2, (list, tuple)) and len(p2) == 2 and
            isinstance(p3, (list, tuple)) and len(p3) == 2):
        return 0.0

    # Calculate the lengths of the sides of the triangle
    a = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
    b = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    c = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
    
    # Law of cosines: c^2 = a^2 + b^2 - 2ab * cos(angle)
    numerator = a**2 + b**2 - c**2
    denominator = 2 * a * b
    
    if denominator == 0:
        return 0.0 # Avoid division by zero for collinear points

    # Clamp the value to the valid range for acos to prevent math domain errors
    cosine_angle = max(-1.0, min(1.0, numerator / denominator))
    
    angle_rad = math.acos(cosine_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def calculate_stride_length(front_foot: Tuple[float, float], back_foot: Tuple[float, float], pixels_per_meter: float) -> float:
    """
    Calculate stride length in meters from front and back foot positions.
    """
    if not all([front_foot, back_foot, pixels_per_meter]):
        return 0.0
    
    pixel_distance = math.sqrt((front_foot[0] - back_foot[0])**2 + (front_foot[1] - back_foot[1])**2)
    return pixel_distance / pixels_per_meter

def calculate_stride_percentage(stride_length: float, bowler_height: float) -> float:
    """
    Calculate stride length as percentage of bowler height.
    Returns percentage and whether it's in optimal range.
    """
    if not all([stride_length, bowler_height]):
        return 0.0
    
    percentage = (stride_length / bowler_height) * 100
    return percentage

def calculate_release_height(release_point: Tuple[float, float], ground_point: Tuple[float, float], pixels_per_meter: float) -> float:
    """
    Calculate release height in meters.
    """
    if not all([release_point, ground_point, pixels_per_meter]):
        return 0.0
    
    pixel_height = abs(release_point[1] - ground_point[1])
    return pixel_height / pixels_per_meter

def calculate_runup_velocity(positions: List[Tuple[int, Tuple[float, float]]], fps: float, pixels_per_meter: float) -> float:
    """
    Calculate run-up velocity in m/s from a list of position timestamps.
    positions: List of (frame_number, (x, y)) tuples
    """
    if len(positions) < 2 or not pixels_per_meter:
        return 0.0
    
    velocities = []
    for i in range(len(positions) - 1):
        frame1, pos1 = positions[i]
        frame2, pos2 = positions[i + 1]
        
        # Calculate time difference
        time_diff = (frame2 - frame1) / fps
        if time_diff <= 0:
            continue
            
        # Calculate distance in pixels then convert to meters
        distance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        distance_meters = distance / pixels_per_meter
        
        # Calculate velocity in m/s
        velocity = distance_meters / time_diff
        velocities.append(velocity)
    
    # Return average velocity if we have measurements
    return np.mean(velocities) if velocities else 0.0

def calculate_arm_rpm(arm_speed_kmh: float, arm_length_meters: float) -> float:
    """
    Convert linear arm speed to RPM using the arm length as radius.
    """
    if not arm_speed_kmh or not arm_length_meters:
        return 0.0
    
    # Convert km/h to m/s
    speed_ms = arm_speed_kmh * (1000 / 3600)
    
    # Calculate angular velocity (ω = v/r)
    angular_velocity = speed_ms / arm_length_meters
    
    # Convert to RPM (revolutions per minute)
    rpm = (angular_velocity * 60) / (2 * math.pi)
    
    return rpm

def calculate_trunk_stability(hip_center: Tuple[float, float], shoulder_center: Tuple[float, float], vertical: Tuple[float, float]) -> float:
    """
    Calculate trunk stability by measuring the angle between the trunk and vertical.
    Returns angle in degrees where 0° is perfectly vertical.
    """
    if not all([hip_center, shoulder_center]):
        return 0.0
    
    return calculate_angle(vertical, hip_center, shoulder_center)

def get_bowling_speed_category(speed_kmh: float) -> str:
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

def analyze_delivery_biomechanics(
    keypoints: Dict,
    release_speed: float,
    pixels_per_meter: float,
    bowler_height: float = 1.8  # Default height if not provided
) -> Dict:
    """
    Comprehensive biomechanical analysis of a bowling delivery.
    Returns a dictionary containing all biomechanical measurements.
    """
    analysis = {
        "speed_kmh": release_speed,
        "speed_category": get_bowling_speed_category(release_speed),
        "biomechanics": {
            "stride_length_m": 0.0,
            "stride_length_percent": 0.0,
            "release_height_m": 0.0,
            "trunk_stability_deg": 0.0,
            "arm_speed_rpm": 0.0,
            "knee_flexion_deg": 0.0,
            "shoulder_angle_deg": 0.0,
            "elbow_angle_deg": 0.0
        },
        "optimal_ranges": {
            "stride_length": "81-90% of height",
            "trunk_stability": "< 10° from vertical",
            "knee_flexion": "15-20° at front foot contact",
            "elbow_angle": "> 150° at release"
        }
    }
    
    # Add actual calculations here based on keypoints
    # This will be populated with real data during video analysis
    
    return analysis 