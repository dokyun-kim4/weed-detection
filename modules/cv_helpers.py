# Helper functions for weed identification subteam

import cv2
import numpy as np
from io import BytesIO, BufferedReader
import copy
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def get_green(orig_img: np.ndarray) -> np.ndarray:
    """
    Given numpy array representation of image, return an image with the green parts isolated.

    Args:
        orig_img: original image in numpy array form (width x height x 3)

    Returns:
        green_areas: numpy array representing the green-isolated image
    """

    # low/high HSV limits
    LOWER_GREEN = np.array([30, 40, 30])
    UPPER_GREEN = np.array([100, 255, 255])

    # Convert the image from BGR to HSV color space
    rgb_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    # Create a mask to isolate the green areas
    mask = cv2.inRange(rgb_image, LOWER_GREEN, UPPER_GREEN)

    # Apply the mask to the original image
    green_areas = cv2.bitwise_and(orig_img, orig_img, mask=mask)

    return green_areas


def green_to_bnw(green_areas_denoised: np.ndarray) -> np.ndarray:
    # TODO
    # Add docstrings
    glayer = green_areas_denoised[:, :, 1]
    bnw = copy.deepcopy(glayer)
    for i in range(bnw.shape[0]):
        for j in range(bnw.shape[1]):
            if bnw[i, j] > 0:
                bnw[i, j] = 255
    return bnw


def binary_to_cartesian(bnw_array: np.ndarray) -> list:
    """
    Given numpy array of 0s and 255s that represent a black and white image (width x height),
    return two lists that have the x and y coordinates of white areas.

    Args:
        colormap: black and white image that only have values [0, 255]

    Returns:
        xys: 2d array that contains [x,y] points of all white areas
    """
    xys = []
    for y, row in enumerate(bnw_array):
        for x, value in enumerate(row):
            if value == 255:
                xys.append([x, abs(y - bnw_array.shape[0])])

    print("added")
    return np.array(xys)


def DBSCAN_clustering(white_points) -> list:
    """
    Given an array with two columns which stores x and y values representing
    white points in the denoised image, return an array where each row
    classifies which cluster a point is in (same amount of rows as the white
    points array)

    PARAMETERS
    ----------
        white_points: list
            arr with each row containing two values representing x and y values

    RETURNS
    -------
        A n by 1 arr
    """
    dbscan_model = DBSCAN(eps=5, min_samples=10)
    dbscan_model.fit(white_points)
    dbscan_result = dbscan_model.fit_predict(white_points)
    return dbscan_result


def find_bounding_boxes(white_points, dbscan_result):
    """
    Find the bounding box for each cluster.

    Parameters:
        white_points (numpy.ndarray): Array of points in the image.
        dbscan_result (numpy.ndarray): Result of DBSCAN clustering algorithm.

    Returns:
        list: List of bounding boxes for each cluster.
    """
    bounding_boxes = []
    for cluster_label in np.unique(dbscan_result):
        cluster_points = white_points[dbscan_result == cluster_label]
        if len(cluster_points) == 0:
            continue  # Skip clusters with no points
        min_x = np.min(cluster_points[:, 0])
        min_y = np.min(cluster_points[:, 1])
        max_x = np.max(cluster_points[:, 0])
        max_y = np.max(cluster_points[:, 1])
        bounding_boxes.append(((min_x, min_y), (max_x, max_y)))
    return bounding_boxes


def find_cluster_centers(white_points, dbscan_result):
    """
    Find the center of each cluster.

    Parameters:
        white_points (numpy.ndarray): Array of points in the image.
        dbscan_result (numpy.ndarray): Result of DBSCAN clustering algorithm.

    Returns:
        list: List of cluster centers.
    """
    cluster_centers = []
    for cluster_label in np.unique(dbscan_result):
        cluster_points = white_points[dbscan_result == cluster_label]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)
    return cluster_centers


def plot_boxes(image_with_boxes, bounding_boxes):
    """
    Plot bounding boxes around each cluster on the image.

    Parameters:
        image (numpy.ndarray): Input image.
        bounding_boxes (list): List of bounding boxes for each cluster.
    """
    for box in bounding_boxes:
        (min_x, min_y), (max_x, max_y) = box
        cv2.rectangle(image_with_boxes, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    return image_with_boxes


def plot_centers(image_with_centers, cluster_centers):
    """
    Plot cluster centers on the image.

    Parameters:
        image (numpy.ndarray): Input image.
        cluster_centers (list): List of cluster centers.
    """
    for center in cluster_centers:
        x, y = center.astype(int)
        cv2.circle(
            image_with_centers, (x, y), radius=5, color=(255, 0, 0), thickness=-1
        )
    return image_with_centers


def return_image_array(box, image):
    """
    Returns an array of an image defined by the specified bounding box.

    Parameters:
        box (tuple): A tuple containing two tuples representing the coordinates of the top-left and bottom-right corners of the bounding box.
        image (numpy.ndarray): The input image from which the sub-array is extracted.

    Returns:
        numpy.ndarray or None: An array of the image defined by the bounding box. Returns None if the box width or height is non-positive.
    """
    (min_x, min_y), (max_x, max_y) = box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if box_width > 0 or box_height > 0:
        return image[min_y:max_y, min_x:max_x]


def arr_to_io_buffered_reader(img_arr):
    """
    Given a numpy arr of an image, return the buffered reader to put into a
    GET request.

    PARAMETERS
    ----------
        img_arr: np.array
            numpy array of image

    RETURNS
    -------
        BufferedReader
    """
    ret, img_encode = cv2.imencode(".jpg", img_arr)
    str_encode = img_encode.tostring()
    img_byteio = BytesIO(str_encode)
    img_byteio.name = "img.jpg"
    reader = BufferedReader(img_byteio)
    return reader
