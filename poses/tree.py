import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

def calculate_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def is_tree_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return image, "No Pose Detected"

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    try:
        # Extract landmarks
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate conditions for Tree Pose
        standing_leg_straight = abs(calculate_angle(left_hip, left_knee, left_ankle)) > 150
        one_raised_leg = left_heel.y < right_heel.y or right_heel.y < left_heel.y  # One leg raised
        shoulders_aligned = abs(left_shoulder.x - right_shoulder.x) < 0.2  # Shoulders aligned
        shoulder_tilt = abs(left_shoulder.y - right_shoulder.y) < 0.2  # Shoulder tilt minimal
        distance = calculate_distance(left_wrist, right_wrist)

        # Debugging prints
        print("Standing Leg Straight:", standing_leg_straight)
        print("One Raised Leg:", one_raised_leg)
        print("Shoulders Aligned:", shoulders_aligned)
        print("Shoulder Tilt:", shoulder_tilt)

        # Check conditions for Tree Pose
        if standing_leg_straight and one_raised_leg and shoulders_aligned and shoulder_tilt and distance < 0.03:
            return image, "Tree Pose Detected"

    except IndexError:
        return image, "Error: Missing landmarks in detection."

    return image, "No Pose Detected"