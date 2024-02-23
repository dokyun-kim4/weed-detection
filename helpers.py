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
    # Convert the image from BGR to HSV color space
    rgb_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the green color in HSV
    lower_green = np.array([30,40,30])
    upper_green = np.array([100,255,255])

    # Create a mask to isolate the green areas
    mask = cv2.inRange(rgb_image, lower_green, upper_green)

    # Apply the mask to the original image
    green_areas = cv2.bitwise_and(orig_img, orig_img, mask=mask)

    return green_areas
