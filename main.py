import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN

# Custom helpers
import modules.cv_helpers as ch
import modules.plantnet_helpers as ph

# Define DBscan model
dbscan_model = DBSCAN(eps=5, min_samples=10)

# Read in image
image = cv.imread("./img/img1.png")

# TODO
# video camera trigger

# Get green areas and remove noise
green_areas = ch.get_green(image)

img_denoised = cv.fastNlMeansDenoisingColored(
    green_areas, None, h=70, templateWindowSize=7, searchWindowSize=21
)
# Convert to binary image to apply clustering algorithm
bnw_image = ch.green_to_bnw(img_denoised)
white_points = ch.binary_to_cartesian(bnw_image)
# TODO
# ----- Perform DBscan -------
clusters = ch.DBSCAN_clustering(white_points)
# This works?
#dbscan_model = DBSCAN(eps=5, min_samples=10)

# train the model
#dbscan_model.fit(white_points)
# assign each data point to a cluster
#clusters = dbscan_model.fit_predict(white_points)
#print(dbscan_result)

# Find bbox and center points

bbox_and_center = [
    (bbox, center)
    for bbox, center in zip(
        ch.find_bounding_boxes(white_points, clusters),
        ch.find_cluster_centers(white_points, clusters),
    )
]

print(bbox_and_center)
# TODO
# Iterate through bbox and center point list, call GET request on plantnet


# TODO
# Return results
