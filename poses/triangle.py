import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=2
)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def are_landmarks_visible(landmarks, threshold=0.5):
    return all(landmark.visibility > threshold for landmark in landmarks)

def is_triangle_pose(image):
    annotated_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return annotated_image, "No Pose Detected"

    try:
        landmarks = results.pose_landmarks.landmark

        # Get key landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate key measurements
        # Leg angles
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Torso angle relative to vertical
        torso_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.NOSE],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        )
        
        # Stance width
        feet_distance = calculate_distance(left_ankle, right_ankle)
        hip_distance = calculate_distance(left_hip, right_hip)
        feet_hip_ratio = feet_distance / hip_distance

        # Arm position
        arm_angle = calculate_angle(left_wrist, left_shoulder, right_shoulder)

        # Updated conditions for Triangle Pose
        conditions = {
            "wide_stance": feet_hip_ratio > 1.5,  # Feet wider than hips
            "legs_straight": min(left_leg_angle, right_leg_angle) > 160,  # Allow slight bend
            "torso_tilt": 45 < torso_angle < 135,  # Side bend
            "arms_extended": arm_angle > 150,  # Arms in line
        }

        # Debug information
        print("\nTriangle Pose Measurements:")
        print(f"Feet-Hip Ratio: {feet_hip_ratio:.2f}")
        print(f"Left Leg Angle: {left_leg_angle:.2f}")
        print(f"Right Leg Angle: {right_leg_angle:.2f}")
        print(f"Torso Angle: {torso_angle:.2f}")
        print(f"Arm Angle: {arm_angle:.2f}")
        print("\nConditions Check:")
        for condition, result in conditions.items():
            print(f"{condition}: {result}")

        # If most conditions are met (allow for some flexibility)
        if sum(conditions.values()) >= len(conditions) - 1:  # Allow one condition to fail
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            return annotated_image, "Triangle Pose Detected"

        return annotated_image, "No Pose Detected"

    except Exception as e:
        print(f"Error in is_triangle_pose: {e}")
        return annotated_image, f"Error: {str(e)}"