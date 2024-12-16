import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)


def check_ankle_hip_alignment(hip_left, hip_right, ankle_left, ankle_right, tolerance=0.05):
    # Check if the x-coordinates of the hips and ankles are aligned (within a tolerance)
    if abs(hip_left.x - hip_right.x) < tolerance and abs(ankle_left.x - ankle_right.x) < tolerance:
        return True
    return False


def is_shoulder_stand_pose(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    # Ensure we always return a tuple (image, result)
    if not results.pose_landmarks:
        return image, "No Pose Detected"

    # Draw landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Extract landmarks
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Calculate shoulder and hip distances
    shoulder_distance = math.dist(
        (left_shoulder.x, left_shoulder.y), (right_shoulder.x, right_shoulder.y)
    )
    hip_distance = math.dist(
        (left_hip.x, left_hip.y), (right_hip.x, right_hip.y)
    )

    # Check if ankles and hips are aligned
    ankle_hip_aligned = check_ankle_hip_alignment(left_hip, right_hip, left_ankle, right_ankle)

    # Check conditions for shoulder stand pose
    if ankle_hip_aligned and left_shoulder.y > left_knee.y:
        return image, "Shoulder Standing Pose Detected"

    # Return if no pose detected
    return image, "No Pose Detected"