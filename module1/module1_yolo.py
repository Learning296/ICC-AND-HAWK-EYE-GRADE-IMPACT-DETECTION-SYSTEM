import cv2
import math
import numpy as np
from collections import deque

# ---------------- KALMAN FILTER ----------------
class BallKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32
        )

        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32
        )

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def update(self, cx=None, cy=None):
        if cx is not None and cy is not None:
            measurement = np.array([[cx], [cy]], np.float32)
            if not self.initialized:
                self.kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
                self.initialized = True
            self.kf.correct(measurement)

        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

# ---------------- UTILS ----------------
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# ---------------- PROPOSAL GENERATOR ----------------
class TemporalProposalGenerator:
    def __init__(self, threshold_px=150, min_frames=3):
        self.threshold = threshold_px
        self.min_frames = min_frames
        self.buffer = deque(maxlen=10)

    def update(self, bat_centers, ball_center):
        if not bat_centers or ball_center is None:
            self.buffer.clear()
            return False, None

        closest_bat = min(bat_centers, key=lambda b: euclidean(b, ball_center))
        d = euclidean(closest_bat, ball_center)
        self.buffer.append(d)

        if len(self.buffer) >= self.min_frames:
            if all(x < self.threshold for x in list(self.buffer)[-self.min_frames:]):
                return True, closest_bat

        return False, None
