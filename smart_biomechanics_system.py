import cv2
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque
import json

class SmartBiomechanicsSystem:
    def __init__(self):
        # Safety thresholds for biomechanical alerts
        self.safety_thresholds = {
            "knee_hyperextension": 180,  # degrees
            "elbow_hyperextension": 190,  # degrees
            "head_movement_threshold": 50,  # pixels
            "unstable_stance_ratio": 0.3,  # stance width ratio
            "excessive_backlift": 2.0,  # meters
            "poor_balance_threshold": 30,  # pixels head movement
        }
        
        # Training prescription database
        self.training_prescriptions = {
            "poor_balance": {
                "issue": "Unstable head position and poor balance",
                "drills": [
                    "Single-leg stance practice (30 seconds each leg)",
                    "Head stability drills with focus on keeping head still",
                    "Balance board exercises",
                    "Yoga tree pose and warrior poses"
                ],
                "strength": [
                    "Core strengthening exercises",
                    "Glute activation drills",
                    "Ankle stability exercises"
                ]
            },
            "knee_hyperextension": {
                "issue": "Knee joint extending beyond safe range",
                "drills": [
                    "Controlled knee flexion exercises",
                    "Squat form practice with proper depth",
                    "Knee stability drills"
                ],
                "strength": [
                    "Quadriceps strengthening",
                    "Hamstring flexibility exercises",
                    "Knee proprioception training"
                ]
            },
            "elbow_hyperextension": {
                "issue": "Elbow joint extending beyond safe range",
                "drills": [
                    "Controlled arm movement drills",
                    "Bat swing with proper elbow control",
                    "Soft ball catching practice"
                ],
                "strength": [
                    "Forearm strengthening",
                    "Tricep flexibility exercises",
                    "Shoulder stability work"
                ]
            },
            "unstable_stance": {
                "issue": "Stance too narrow or unstable",
                "drills": [
                    "Stance width practice with markers",
                    "Balance exercises in batting stance",
                    "Foot positioning drills"
                ],
                "strength": [
                    "Leg strength training",
                    "Core stability exercises",
                    "Hip mobility work"
                ]
            },
            "excessive_backlift": {
                "issue": "Bat backlift too high, affecting timing",
                "drills": [
                    "Controlled backlift practice",
                    "Shadow batting with focus on bat path",
                    "Timing drills with lower backlift"
                ],
                "strength": [
                    "Shoulder flexibility exercises",
                    "Upper back strength training",
                    "Wrist and forearm work"
                ]
            }
        }
        
        self.alerts = []
        self.recommendations = []
        self.overlay_data = {}
        self.biomechanics_history = []
        self.quality_scores = []
        
    def analyze_biomechanics_safety(self, keypoints: Dict, biomechanics_data: Dict, frame_count: int) -> List[str]:
        """Analyzes biomechanics for safety issues and returns alerts."""
        alerts = []
        
        if not keypoints:
            return alerts
            
        # Check knee angles (keypoints 11, 13, 15 for left leg)
        if all(k in keypoints for k in [11, 13, 15]):
            knee_angle = self.calculate_angle(keypoints[11], keypoints[13], keypoints[15])
            if knee_angle > self.safety_thresholds["knee_hyperextension"]:
                alert_msg = f"ALERT: Knee hyperextension detected ({knee_angle:.1f}°) at frame {frame_count}"
                alerts.append(alert_msg)
                if alert_msg not in self.alerts:
                    self.alerts.append(alert_msg)
                
        # Check elbow angles (keypoints 5, 7, 9 for left arm)
        if all(k in keypoints for k in [5, 7, 9]):
            elbow_angle = self.calculate_angle(keypoints[5], keypoints[7], keypoints[9])
            if elbow_angle > self.safety_thresholds["elbow_hyperextension"]:
                alert_msg = f"ALERT: Elbow hyperextension detected ({elbow_angle:.1f}°) at frame {frame_count}"
                alerts.append(alert_msg)
                if alert_msg not in self.alerts:
                    self.alerts.append(alert_msg)
        
        # Check head stability
        if "head_stability" in biomechanics_data:
            if biomechanics_data["head_stability"] > self.safety_thresholds["head_movement_threshold"]:
                alert_msg = f"ALERT: Excessive head movement detected ({biomechanics_data['head_stability']:.1f} px) at frame {frame_count}"
                alerts.append(alert_msg)
                if alert_msg not in self.alerts:
                    self.alerts.append(alert_msg)
        
        # Check stance stability
        if "stance_width_ratio" in biomechanics_data:
            if biomechanics_data["stance_width_ratio"] < self.safety_thresholds["unstable_stance_ratio"]:
                alert_msg = f"ALERT: Unstable stance detected (ratio: {biomechanics_data['stance_width_ratio']:.2f}) at frame {frame_count}"
                alerts.append(alert_msg)
                if alert_msg not in self.alerts:
                    self.alerts.append(alert_msg)
        
        # Check backlift height
        if "backlift_height" in biomechanics_data:
            if biomechanics_data["backlift_height"] > self.safety_thresholds["excessive_backlift"]:
                alert_msg = f"ALERT: Excessive backlift detected ({biomechanics_data['backlift_height']:.2f} m) at frame {frame_count}"
                alerts.append(alert_msg)
                if alert_msg not in self.alerts:
                    self.alerts.append(alert_msg)
        
        return alerts
    
    def generate_training_prescriptions(self, biomechanics_data: Dict, alerts: List[str]) -> Dict:
        """Generates training prescriptions based on detected issues."""
        prescriptions = {}
        
        for alert in alerts:
            if "head movement" in alert.lower():
                prescriptions["poor_balance"] = self.training_prescriptions["poor_balance"]
                if "poor_balance" not in [rec.get("issue_type") for rec in self.recommendations]:
                    self.recommendations.append({
                        "issue_type": "poor_balance",
                        "issue": "Unstable head position and poor balance",
                        "drills": self.training_prescriptions["poor_balance"]["drills"][:3],
                        "strength": self.training_prescriptions["poor_balance"]["strength"][:2]
                    })
            elif "knee hyperextension" in alert.lower():
                prescriptions["knee_hyperextension"] = self.training_prescriptions["knee_hyperextension"]
                if "knee_hyperextension" not in [rec.get("issue_type") for rec in self.recommendations]:
                    self.recommendations.append({
                        "issue_type": "knee_hyperextension",
                        "issue": "Knee joint extending beyond safe range",
                        "drills": self.training_prescriptions["knee_hyperextension"]["drills"][:3],
                        "strength": self.training_prescriptions["knee_hyperextension"]["strength"][:2]
                    })
            elif "elbow hyperextension" in alert.lower():
                prescriptions["elbow_hyperextension"] = self.training_prescriptions["elbow_hyperextension"]
                if "elbow_hyperextension" not in [rec.get("issue_type") for rec in self.recommendations]:
                    self.recommendations.append({
                        "issue_type": "elbow_hyperextension",
                        "issue": "Elbow joint extending beyond safe range",
                        "drills": self.training_prescriptions["elbow_hyperextension"]["drills"][:3],
                        "strength": self.training_prescriptions["elbow_hyperextension"]["strength"][:2]
                    })
            elif "unstable stance" in alert.lower():
                prescriptions["unstable_stance"] = self.training_prescriptions["unstable_stance"]
                if "unstable_stance" not in [rec.get("issue_type") for rec in self.recommendations]:
                    self.recommendations.append({
                        "issue_type": "unstable_stance",
                        "issue": "Stance too narrow or unstable",
                        "drills": self.training_prescriptions["unstable_stance"]["drills"][:3],
                        "strength": self.training_prescriptions["unstable_stance"]["strength"][:2]
                    })
            elif "excessive backlift" in alert.lower():
                prescriptions["excessive_backlift"] = self.training_prescriptions["excessive_backlift"]
                if "excessive_backlift" not in [rec.get("issue_type") for rec in self.recommendations]:
                    self.recommendations.append({
                        "issue_type": "excessive_backlift",
                        "issue": "Bat backlift too high, affecting timing",
                        "drills": self.training_prescriptions["excessive_backlift"]["drills"][:3],
                        "strength": self.training_prescriptions["excessive_backlift"]["strength"][:2]
                    })
        
        return prescriptions
    
    def draw_ai_overlays(self, frame: np.ndarray, keypoints: Dict, biomechanics_data: Dict, 
                        alerts: List[str], frame_count: int) -> np.ndarray:
        """Draws AI overlays on the frame showing joint angles, measurements, and alerts."""
        overlay_frame = frame.copy()
        
        if not keypoints:
            return overlay_frame
            
        # Draw joint angles
        self.draw_joint_angles(overlay_frame, keypoints)
        
        # Draw measurements
        self.draw_measurements(overlay_frame, biomechanics_data, keypoints)
        
        # Draw alerts
        self.draw_alerts(overlay_frame, alerts)
        
        # Draw biomechanical indicators
        self.draw_biomechanical_indicators(overlay_frame, biomechanics_data, keypoints)
        
        return overlay_frame
    
    def draw_joint_angles(self, frame: np.ndarray, keypoints: Dict):
        """Draws joint angles on the frame."""
        # Knee angles
        if all(k in keypoints for k in [11, 13, 15]):  # Left knee
            angle = self.calculate_angle(keypoints[11], keypoints[13], keypoints[15])
            mid_point = keypoints[13]
            cv2.putText(frame, f"Knee: {angle:.1f}°", 
                       (int(mid_point[0]) + 10, int(mid_point[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if all(k in keypoints for k in [12, 14, 16]):  # Right knee
            angle = self.calculate_angle(keypoints[12], keypoints[14], keypoints[16])
            mid_point = keypoints[14]
            cv2.putText(frame, f"Knee: {angle:.1f}°", 
                       (int(mid_point[0]) + 10, int(mid_point[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Elbow angles
        if all(k in keypoints for k in [5, 7, 9]):  # Left elbow
            angle = self.calculate_angle(keypoints[5], keypoints[7], keypoints[9])
            mid_point = keypoints[7]
            cv2.putText(frame, f"Elbow: {angle:.1f}°", 
                       (int(mid_point[0]) + 10, int(mid_point[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    def draw_measurements(self, frame: np.ndarray, biomechanics_data: Dict, keypoints: Dict):
        """Draws biomechanical measurements on the frame."""
        # Stance width
        if "stance_width_ratio" in biomechanics_data and all(k in keypoints for k in [15, 16]):
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            mid_x = (left_ankle[0] + right_ankle[0]) / 2
            mid_y = (left_ankle[1] + right_ankle[1]) / 2
            cv2.putText(frame, f"Stance: {biomechanics_data['stance_width_ratio']:.2f}", 
                       (int(mid_x), int(mid_y) + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Backlift height
        if "backlift_height" in biomechanics_data and 0 in keypoints:  # nose keypoint
            nose = keypoints[0]
            cv2.putText(frame, f"Backlift: {biomechanics_data['backlift_height']:.2f}m", 
                       (int(nose[0]) - 50, int(nose[1]) - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def draw_alerts(self, frame: np.ndarray, alerts: List[str]):
        """Draws safety alerts on the frame."""
        y_pos = 50
        for i, alert in enumerate(alerts[:3]):  # Show max 3 alerts
            # Red background for alert
            text_size = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (10, y_pos - 25), (10 + text_size[0] + 10, y_pos + 5), (0, 0, 255), -1)
            cv2.putText(frame, alert, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 40
    
    def draw_biomechanical_indicators(self, frame: np.ndarray, biomechanics_data: Dict, keypoints: Dict):
        """Draws visual indicators for biomechanical quality."""
        # Head stability indicator
        if "head_stability" in biomechanics_data and 0 in keypoints:
            nose = keypoints[0]
            stability = biomechanics_data["head_stability"]
            if stability < 20:
                color = (0, 255, 0)  # Green for good stability
                text = "Stable"
            elif stability < 40:
                color = (0, 255, 255)  # Yellow for moderate
                text = "Moderate"
            else:
                color = (0, 0, 255)  # Red for poor
                text = "Unstable"
            
            cv2.putText(frame, f"Head: {text}", 
                       (int(nose[0]) - 30, int(nose[1]) - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Calculates angle between three points."""
        if not all(p for p in [p1, p2, p3]):
            return 0.0
        
        a = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
        b = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        c = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
        
        if a * b == 0:
            return 0.0
            
        cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))
    
    def draw_smart_sidebar(self, frame: np.ndarray, biomechanics_data: Dict, alerts: List[str], 
                          prescriptions: Dict, frame_count: int) -> np.ndarray:
        """Draws a smart sidebar with alerts and training recommendations."""
        h, w = frame.shape[:2]
        sidebar_width = 450
        sidebar = np.zeros((h, sidebar_width, 3), dtype=np.uint8)
        
        y_pos = 30
        
        # Title
        cv2.putText(sidebar, "SMART BIOMECHANICS", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        y_pos += 40
        
        # Alerts Section
        cv2.putText(sidebar, "SAFETY ALERTS:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_pos += 30
        
        if alerts:
            for alert in alerts[:3]:  # Show max 3 alerts
                cv2.putText(sidebar, f"• {alert}", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += 25
        else:
            cv2.putText(sidebar, "✓ No safety issues detected", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += 25
        
        y_pos += 20
        
        # Training Recommendations
        cv2.putText(sidebar, "TRAINING RECOMMENDATIONS:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_pos += 30
        
        if prescriptions:
            for issue, prescription in prescriptions.items():
                cv2.putText(sidebar, f"• {prescription['issue']}", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += 20
                cv2.putText(sidebar, "  Drills:", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 20
                for drill in prescription['drills'][:2]:  # Show first 2 drills
                    cv2.putText(sidebar, f"    - {drill}", (35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                    y_pos += 15
                y_pos += 10
        else:
            cv2.putText(sidebar, "✓ Good biomechanics detected", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += 25
        
        y_pos += 20
        
        # Live Biomechanics Quality Score
        quality_score = self.calculate_biomechanics_quality_score(biomechanics_data, alerts)
        cv2.putText(sidebar, f"BIOMECHANICS QUALITY: {quality_score:.0f}%", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return np.hstack((frame, sidebar))
    
    def calculate_biomechanics_quality_score(self, biomechanics_data: Dict, alerts: List[str]) -> float:
        """Calculates overall biomechanics quality score (0-100)."""
        base_score = 100.0
        
        # Deduct points for alerts
        base_score -= len(alerts) * 15
        
        # Deduct points for poor biomechanics
        if "head_stability" in biomechanics_data:
            if biomechanics_data["head_stability"] > 50:
                base_score -= 20
            elif biomechanics_data["head_stability"] > 30:
                base_score -= 10
        
        if "stance_width_ratio" in biomechanics_data:
            if biomechanics_data["stance_width_ratio"] < 0.3:
                base_score -= 15
        
        quality_score = max(0.0, base_score)
        
        # Store quality score history
        self.quality_scores.append(quality_score)
        
        return quality_score
    
    def get_analysis_summary(self) -> Dict:
        """Returns a summary of the smart analysis."""
        avg_quality = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0
        
        return {
            "total_alerts": len(self.alerts),
            "alerts": self.alerts,
            "recommendations": self.recommendations,
            "overlay_data": self.overlay_data,
            "quality_analysis": {
                "average_quality_score": round(avg_quality, 2),
                "min_quality_score": min(self.quality_scores) if self.quality_scores else 0,
                "max_quality_score": max(self.quality_scores) if self.quality_scores else 0,
                "total_frames_analyzed": len(self.quality_scores)
            }
        }


