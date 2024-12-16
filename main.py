import cv2
from poses.chair import is_chair_pose
from poses.warrior import is_warrior_pose
from poses.cobra import is_cobra_pose
from poses.dog import is_dog_pose
from poses.shoulder_stand import is_shoulder_stand_pose
from poses.tree import is_tree_pose
from poses.triangle import is_triangle_pose
def detect_pose(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Image could not be loaded. Check the path."

    # Try detecting chair pose
    chair_image, chair_result = is_chair_pose(image)

    if "Chair Pose Detected" in chair_result:
        return chair_image, chair_result

    # Try detecting warrior pose
    warrior_image, warrior_result = is_warrior_pose(image)
    if "Warrior Pose Detected" in warrior_result:
        return warrior_image, warrior_result
    

    cobra_image, cobra_result = is_cobra_pose(image)
    if "Cobra Pose Detected" in cobra_result:
        return cobra_image, cobra_result
    
    


    shoulder_image, shoulder_result = is_shoulder_stand_pose(image)
    if "Shoulder Standing Pose Detected" in shoulder_result:
        return shoulder_image, shoulder_result
    
    dog_image, dog_result = is_dog_pose(image)
    if "Dog Pose Detected" in dog_result:
        return dog_image, dog_result
    
    tree_image, tree_result = is_tree_pose(image)
    if "Tree Pose Detected" in tree_result:
        return tree_image, tree_result
    

    triangle_image, triangle_result = is_triangle_pose(image)
    if "Triangle Pose Detected" in triangle_result:
        return triangle_image, triangle_result

    return image, "No Pose Detected"

    

if __name__ == "__main__":
    # Specify the path to the image
    image_path = "dog.jpg"  # Adjust this path as needed

    # Detect the pose
    processed_image, result = detect_pose(image_path)

    # Display the resultq
    if processed_image is not None:
        cv2.putText(processed_image, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if "Detected" in result else (0, 0, 255), 2)
        cv2.imshow("Yoga Pose Detection", processed_image)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    else:
        print(result)
