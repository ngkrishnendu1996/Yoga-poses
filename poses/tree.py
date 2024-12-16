import cv2
import mediapipe as mp
import math

mp_pose=mp.solutions.pose
pose_detector=mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

def is_tree_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    # Add this early return to handle cases with no detected landmarks
    if not results.pose_landmarks:
        return image, "No Pose Detected"

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Modify the standing leg condition
    standing_leg_straight = abs(calculate_angle(left_hip, left_knee, left_ankle)) > 160

    # Check if one foot is raised and near the opposite inner thigh or calf
    raised_leg_condition = (
        abs(left_ankle.x - right_knee.x) < 0.1 or abs(right_ankle.x - left_knee.x) < 0.1
    )

    # Ensure the upper body remains straight
    shoulders_aligned = abs(left_shoulder.x - right_shoulder.x) < 0.1

    if standing_leg_straight and raised_leg_condition and shoulders_aligned:
        return image, "Tree Pose Detected"
    
    return image, "No Pose Detected"