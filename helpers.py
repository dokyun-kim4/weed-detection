import cv2
import numpy as np

def get_green(orig_img: np.ndarray) -> np.ndarray:
    """
    Given numpy array representation of image, return an image with the green parts isolated.

    Args:
        orig_img: original image in numpy array form
    
    Returns:
        green_areas: numpy array representing the green-isolated image
    """

    # low/high HSV limits
    LOWER_GREEN = np.array([30,40,30])
    UPPER_GREEN = np.array([100,255,255])

    # Convert the image from BGR to HSV color space
    rgb_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    # Create a mask to isolate the green areas
    mask = cv2.inRange(rgb_image, LOWER_GREEN, UPPER_GREEN)

    # Apply the mask to the original image
    green_areas = cv2.bitwise_and(orig_img, orig_img, mask=mask)

    return green_areas
