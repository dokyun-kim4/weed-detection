import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import time
from dotenv import load_dotenv

# Custom helpers
import modules.cv_helpers as ch
import modules.plantnet_helpers as ph

SAFE_PLANTS = ["Lettuce"]

load_dotenv()
endpoint = ph.API_Initialize(os.getenv("2b109rJwhiLaSgff3H6wVCoT3u"))

# Read in image
image = cv.imread("./img/lettuce_test.png")
img_copy = np.copy(image)

# TODO
# video camera trigger
tic = time.perf_counter()
print("Denoise Started")
# Get green areas and remove noise
green_areas = ch.get_green(image)
bnw_image = ch.green_to_bnw(green_areas)

# img_denoised = cv.fastNlMeansDenoisingColored(
#     bnw_image, None, h=100, templateWindowSize=7, searchWindowSize=21
# )
img_denoised = cv.GaussianBlur(bnw_image, (13, 13), 0)
# Convert to binary image to apply clustering algorithm
img_denoised = np.mean(img_denoised, axis=2)
img_denoised[img_denoised > 150] = 255
img_denoised[img_denoised < 150] = 0

# Reduce resolution of image and find white points
low_res_bnw_image, old_new_image_ratio = ch.refactor_to_lower_res(img_denoised)
white_points_low_res = ch.binary_to_cartesian(low_res_bnw_image)
white_points = white_points_low_res * old_new_image_ratio
white_points = white_points.round().astype(int)

toc = time.perf_counter()
print(f"Denoise Complete: {toc-tic:0.4f} seconds, Starting Clustering")
tic = time.perf_counter()
# ----- Perform DBscan -------
clusters = ch.DBSCAN_clustering(white_points_low_res)
toc = time.perf_counter()
print(f"Clustering Complete: {toc-tic:0.4f} seconds, Starting Bbox Locating")
# Find bbox and center points
tic = time.perf_counter()
bboxes = [(bbox[0], bbox[1]) for bbox in ch.find_bounding_boxes(white_points, clusters)]
centers = [center for center in ch.find_cluster_centers(white_points, clusters)]
toc = time.perf_counter()
print(f"Bounding Boxes Located: {toc-tic:0.4f} seconds, Starting API Request")
# Iterate through bbox list, call GET request on plantnet
print("Requesting Pl@ntNet API")
tic = time.perf_counter()
for bbox in bboxes:
    segmented_img = ch.return_image_array(bbox, img_copy, min_size=10000)
    if segmented_img is None:
        print("plant not found")
    else:
        img_buffer = ch.arr_to_io_buffered_reader(segmented_img)
        data, files = ph.load_plant_data(img_buffer, Organ=["leaf"])
        result = ph.Send_API_Request(endpoint, files, data)
        print(result["results"][0]["species"]["commonNames"])
        print(bbox)
toc = time.perf_counter()
print(f"Request Returned {toc-tic:0.4f} seconds")