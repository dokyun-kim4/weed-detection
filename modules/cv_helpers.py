# Helper functions for weed identification subteam

import cv2
import numpy as np
from io import BytesIO, BufferedReader

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

def arr_to_io_buffered_reader(img_arr):
    ret, img_encode = cv2.imencode('.jpg', img_arr)
    str_encode = img_encode.tostring()
    img_byteio = BytesIO(str_encode)
    img_byteio.name = 'img.jpg'
    reader = BufferedReader(img_byteio)
    return reader