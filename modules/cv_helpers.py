# Helper functions for weed identification subteam

import cv2
import numpy as np
from scipy.stats import moment

def get_green(orig_img: np.ndarray) -> np.ndarray:
    """
    Given numpy array representation of image, return an image with the green parts isolated.

    Args:
        orig_img: original image in numpy array form (width x height x 3)
    
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

def binary_to_cartesian(bnw_array: np.ndarray) -> list:
    """
    Given numpy array of 0s and 255s that represent a black and white image (width x height), 
    return two lists that have the x and y coordinates of white areas.

    Args:
        colormap: black and white image that only have values [0, 255]

    Returns:
        xs: list of x coordinates of black areas
        ys: list of y coordinates of white areas
    """
    xs,ys = [],[]
    for y, row in enumerate(bnw_array):
        for x, value in enumerate(row):
            if value == 255:
                xs.append(x)
                ys.append(abs(y-bnw_array.shape[0]))
    return xs,ys

def find_centroid_of_blob(x: list, y: list):
    """
    Given a list of x and y coordinates of white points, return the centroid 
    of the white blob

    PARAMETERS
    ----------
        x: list
            list of x-coords
        y: list
            list of y-coords

    RETURNS
    -------
        Returns the centroid coordinates as a tuple: (x_coords, y_coords)
    """
    M00 = len(x)
    M10 = sum(x)
    M01 = sum(y)
    return M10/M00, M01/M00