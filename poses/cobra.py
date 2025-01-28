import cv2
import mediapipe as mp
import math

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (a, b, c).
    """
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    return abs(angle)

def is_cobra_pose(image):
    """
    Detects if the given pose is a Cobra Pose based on specific measurements.
    Also checks if feet, knees, and palms are on the floor.
    """
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return image, "No Pose Detected"

    # Draw landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Extract key landmarks
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    right_ear=results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
    right_pinky = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]

    # Calculate shoulder-hip distances
    left_shoulder_hip_distance = abs(left_shoulder.y - left_hip.y)
    right_shoulder_hip_distance = abs(right_shoulder.y - right_hip.y)

    # Check if elbows are bent
    left_elbow_bent = abs(left_elbow.y - left_wrist.y) < abs(left_shoulder.y - left_elbow.y)
    right_elbow_bent = abs(right_elbow.y - right_wrist.y) < abs(right_shoulder.y - right_elbow.y)

    # Check if hips are grounded
    hips_grounded = left_hip.y > left_shoulder.y and right_hip.y > right_shoulder.y

    # Check if feet are on the floor
    feet_on_floor = (
        abs(left_ankle.y - right_ankle.y) < 0.05  # Feet should be roughly aligned
    )

    # Check if knees are on the floor
    knees_on_floor = (
        abs(left_knee.y - right_knee.y) < 0.03  # Knees should be roughly aligned
    )

    # Check if palms are on the floor
    palms_on_floor = (
        abs(left_wrist.y - right_wrist.y) < 0.05  # Wrists should be roughly aligned
        and left_wrist.y > left_elbow.y  # Ensure wrists are below elbows
        and right_wrist.y > right_elbow.y
    )
    shoulder_hip_knee=calculate_angle(left_shoulder,left_hip,left_knee)

    head_align=abs(right_ear.x - right_pinky.x)

    # Determine if the Cobra Pose criteria are met
    if (
        left_shoulder_hip_distance < 0.2 and right_shoulder_hip_distance < 0.3  # Adjusted thresholds
        and left_elbow_bent and right_elbow_bent
        and hips_grounded
        and feet_on_floor  and palms_on_floor
        and shoulder_hip_knee<230 and knees_on_floor
        and head_align < 0.1
    ):
        return image, "Cobra Pose Detected"

    return image, "No Pose Detected"
