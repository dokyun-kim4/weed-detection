import cv2 as cv
import time

# Custom helpers
import modules.cv_helpers as ch
import modules.plantnet_helpers as ph

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
low_res_bnw_image, old_new_image_ratio = ch.refactor_to_lower_res(bnw_image)
white_points_low_res = ch.binary_to_cartesian(low_res_bnw_image)
white_points = white_points_low_res * old_new_image_ratio
white_points = white_points.round()

# ----- Perform DBscan -------
start = time.time()
clusters = ch.DBSCAN_clustering(white_points)
print("--- %s seconds ---" % (time.time() - start))

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
