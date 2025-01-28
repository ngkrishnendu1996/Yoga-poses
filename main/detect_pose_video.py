import cv2
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import pose detection functions from the poses module
from poses.chair import is_chair_pose
from poses.warrior import is_warrior_pose
from poses.cobra import is_cobra_pose
from poses.dog import is_dog_pose
from poses.shoulder_stand import is_shoulder_stand_pose
from poses.tree import is_tree_pose
from poses.triangle import is_triangle_pose


def detect_pose(frame):
    """
    Detect yoga poses in a single frame.

    Args:
        frame (numpy.ndarray): Frame from webcam video.

    Returns:
        tuple: Processed frame and detection result.
    """
    if frame is None:
        return None, "Error: Invalid frame"

    # Create copies for each pose detection to prevent landmark accumulation
    poses_to_check = [
        (lambda img: is_chair_pose(img.copy()), "Chair Pose Detected"),
        (lambda img: is_warrior_pose(img.copy()), "Warrior Pose Detected"),
        (lambda img: is_cobra_pose(img.copy()), "Cobra Pose Detected"),
        (lambda img: is_shoulder_stand_pose(img.copy()), "Shoulder Standing Pose Detected"),
        (lambda img: is_dog_pose(img.copy()), "Dog Pose Detected"),
        (lambda img: is_tree_pose(img.copy()), "Tree Pose Detected"),
        (lambda img: is_triangle_pose(img.copy()), "Triangle Pose Detected"),
    ]

    # Check each pose with a fresh copy of the frame
    debug_info = []  # Collect debug information
    for pose_func, pose_name in poses_to_check:
        try:
            processed_frame, result = pose_func(frame)
            debug_info.append(f"{pose_name}: {result}")
            if pose_name in result:
                return processed_frame, result
        except Exception as e:
            debug_info.append(f"Error in {pose_name}: {e}")
            continue

    # Print debug information
    print("\nPose Detection Debug Info:")
    for info in debug_info:
        print(info)

    # If no pose is detected, return the original frame
    return frame, "No Pose Detected"


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Open the default webcam

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        exit()

    print("Press 'q' to quit the application.")
    print("Press 'd' to toggle debug information.")
    show_debug = False

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break

            # Detect pose in the current frame
            processed_frame, result = detect_pose(frame)

            # Display the result on the frame
            cv2.putText(processed_frame, result, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 255, 0) if "Detected" in result else (0, 0, 255), 
                        2)

            # Display debug information if enabled
            if show_debug:
                cv2.putText(processed_frame, "Debug Mode ON", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Show the frame
            cv2.imshow("Yoga Pose Detection", processed_frame)

            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit application
                break
            elif key == ord('d'):  # Toggle debug mode
                show_debug = not show_debug
                print(f"\nDebug mode {'enabled' if show_debug else 'disabled'}")

        except Exception as e:
            print(f"Error in main loop: {e}")
            continue

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
