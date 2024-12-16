import cv2
import mediapipe as mp
import math

mp_pose=mp.solutions.pose
pose_detector=mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

def is_dog_pose(image):
    """Detects if the image contains a dog pose."""
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    # Ensure we always return a tuple (image, result)
    if not results.pose_landmarks:
        return image, "No Pose Detected"

    # Draw pose landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract relevant landmarks
    landmarks = results.pose_landmarks.landmark
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    nose=landmarks[mp_pose.PoseLandmark.NOSE]
    left_pinky=landmarks[mp_pose.PoseLandmark.LEFT_PINKY]
    left_heel=landmarks[mp_pose.PoseLandmark.LEFT_HEEL]

            # Calculate conditions
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
    legs_straight = (left_leg_angle > 160 and right_leg_angle > 160)

    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    arms_straight = (left_arm_angle > 160 and right_arm_angle > 160)

    hips_above_shoulders=left_shoulder.y>left_hip.y

    head_below_shoulder=nose.y> left_shoulder.y

            
            

            # Check dog pose conditions
    if hips_above_shoulders and arms_straight and legs_straight and head_below_shoulder and left_pinky.x <left_heel.x:
            return image, "Dog Pose Detected"

            # Fallback
    return image, "No Pose Detected"






