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
    LOWER_GREEN = np.array([30, 40, 30])
    UPPER_GREEN = np.array([100, 255, 255])

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
    xs, ys = [], []
    for y, row in enumerate(bnw_array):
        for x, value in enumerate(row):
            if value == 255:
                xs.append(x)
                ys.append(abs(y - bnw_array.shape[0]))
    return xs, ys


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
    return M10 / M00, M01 / M00


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


def plot_boxes(image, bounding_boxes):
    """
    Plot bounding boxes around each cluster on the image.

    Parameters:
        image (numpy.ndarray): Input image.
        bounding_boxes (list): List of bounding boxes for each cluster.
    """
    for box in bounding_boxes:
        (min_x, min_y), (max_x, max_y) = box
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    return image


def plot_centers(image, cluster_centers):
    """
    Plot cluster centers on the image.

    Parameters:
        image (numpy.ndarray): Input image.
        cluster_centers (list): List of cluster centers.
    """
    for center in cluster_centers:
        x, y = center.astype(int)
        cv2.circle(image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    return image
