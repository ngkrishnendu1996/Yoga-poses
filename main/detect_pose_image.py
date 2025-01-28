import cv2
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from poses.chair import is_chair_pose
from poses.warrior import is_warrior_pose
from poses.cobra import is_cobra_pose
from poses.dog import is_dog_pose
from poses.shoulder_stand import is_shoulder_stand_pose
from poses.tree import is_tree_pose
from poses.triangle import is_triangle_pose

def detect_pose(image_path):
    # Load the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        return None, "Image could not be loaded. Check the path."

    # Create copies for each pose detection to prevent landmark accumulation
    poses_to_check = [
        (lambda img: is_chair_pose(img.copy()), "Chair Pose Detected"),
        (lambda img: is_warrior_pose(img.copy()), "Warrior Pose Detected"),
        (lambda img: is_cobra_pose(img.copy()), "Cobra Pose Detected"),
        (lambda img: is_shoulder_stand_pose(img.copy()), "Shoulder Standing Pose Detected"),
        (lambda img: is_dog_pose(img.copy()), "Dog Pose Detected"),
        (lambda img: is_tree_pose(img.copy()), "Tree Pose Detected"),
        (lambda img: is_triangle_pose(img.copy()), "Triangle Pose Detected")
    ]

    # Check each pose with a fresh copy of the image
    for pose_func, pose_name in poses_to_check:
        processed_image, result = pose_func(original_image)
        if pose_name in result:
            return processed_image, result

    # If no pose is detected, return the original image
    return original_image, "No Pose Detected"

if __name__ == "__main__":
    # Specify the path to the image
    image_path = "cobra.jpg"  # Adjust this path as needed

    # Detect the pose
    processed_image, result = detect_pose(image_path)

    # Display the result
    if processed_image is not None:
        # Create a clean copy for final display
        display_image = processed_image.copy()
        
        # Add the result text
        cv2.putText(display_image, result, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if "Detected" in result else (0, 0, 255), 
                    2)
        
        # Show the image
        cv2.imshow("Yoga Pose Detection", display_image)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    else:
        print(result)