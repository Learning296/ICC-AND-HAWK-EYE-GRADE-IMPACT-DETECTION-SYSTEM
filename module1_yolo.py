# module1_yolo.py
import math

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def generate_impact_proposal(
    bat_centers,
    ball_centers,
    distance_threshold_px=150
):
    """
    PURE PROPOSAL GENERATOR
    - NO confirmation
    - NO speed
    - NO cooldown
    """
    if not bat_centers or not ball_centers:
        return False, None

    min_dist = 1e9
    best_loc = None

    for b in bat_centers:
        for c in ball_centers:
            d = euclidean(b, c)
            if d < min_dist:
                min_dist = d
                best_loc = b

    if min_dist < distance_threshold_px:
        return True, best_loc  # MAYBE impact

    return False, None
