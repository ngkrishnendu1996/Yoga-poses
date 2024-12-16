import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

def is_chair_pose(image):
    # Convert the image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    # Check for pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        # Extract relevant landmarks
        hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee_left = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle_left = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        wrist_left = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # Calculate angles
        knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)

        # Pose Detection Logic
        is_knee_correct = knee_angle_left > 90 and hip_left.y < knee_left.y
        is_hand_raised = wrist_left.y < shoulder_left.y

        if is_knee_correct and is_hand_raised:
            return image, "Chair Pose Detected"

    return image, "No Pose Detected"
