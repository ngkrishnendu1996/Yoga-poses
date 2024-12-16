import cv2
import mediapipe as mp
import math

mp_pose=mp.solutions.pose
pose_detector=mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

def is_triangle_pose(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

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
    left_pinky = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY]
    left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    nose=results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)

    # Check leg straightness
    leg_straight = abs(left_knee_angle - 180) < 20

    # Check torso alignment (shoulder tilt for triangle pose)
    torso_tilted = shoulder_tilt > 0.1  # Adjust tolerance if needed

    # Check diagonal alignment of arms (optional for stricter detection)
    left_arm_angle = calculate_angle(left_wrist, left_shoulder, right_shoulder)
    right_arm_angle = calculate_angle(right_wrist, right_shoulder, left_shoulder)

    arms_diagonal = (170 <= left_arm_angle <= 190) or (170 <= right_arm_angle <= 190)

    if leg_straight and torso_tilted and arms_diagonal:
        return image, "Triangle Pose Detected"
    return image,"No Pose Detected"
    

    

    



    
       









