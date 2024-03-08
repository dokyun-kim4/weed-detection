import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Custom helpers
import modules.cv_helpers as ch
import modules.plantnet_helpers as ph

SAFE_PLANTS = ["Lettuce"]

load_dotenv()
endpoint = ph.API_Initialize(os.getenv("KEY"))

# Define DBscan model
dbscan_model = DBSCAN(eps=5, min_samples=10)

# Read in image
image = cv.imread("./img/lettuce_test.png")
img_copy = np.copy(image)

# TODO
# video camera trigger

# Get green areas and remove noise
green_areas = ch.get_green(image)
bnw_image = ch.green_to_bnw(green_areas)

img_denoised = cv.fastNlMeansDenoisingColored(
    bnw_image, None, h=100, templateWindowSize=7, searchWindowSize=21
)
# Convert to binary image to apply clustering algorithm
img_denoised = np.mean(img_denoised, axis=2)
img_denoised[img_denoised > 150] = 255
img_denoised[img_denoised < 150] = 0

white_points = ch.binary_to_cartesian(img_denoised)

# ----- Perform DBscan -------
clusters = ch.DBSCAN_clustering(white_points)

# Find bbox and center points
bboxes = [(bbox[0], bbox[1]) for bbox in ch.find_bounding_boxes(white_points, clusters)]
centers = [center for center in ch.find_cluster_centers(white_points, clusters)]

# TODO
# Iterate through bbox list, call GET request on plantnet
for bbox in bboxes:
    segmented_img = ch.return_image_array(bbox, img_copy)
    img_buffer = ch.arr_to_io_buffered_reader(img_copy)
    data, files = ph.load_plant_data(img_buffer, Organ=["leaf"])
    result = ph.Send_API_Request(endpoint, files, data)
    print(result["results"][0]["species"]["commonNames"])
