import cv2 as cv
from sklearn.cluster import DBSCAN

# Custom helpers
import modules.cv_helpers as ch
import modules.plantnet_helpers as ph

# Define DBscan model
dbscan_model = DBSCAN(eps=5, min_samples=10)

# Read in image
image= cv.imread('img/img1.png')

# TODO
# video camera trigger

# Get green areas and remove noise
green_areas = ch.get_green(image)
img_denoised = cv.fastNlMeansDenoisingColored(green_areas, None, h=70, templateWindowSize=7, searchWindowSize=21)

# Convert to binary image to apply clustering algorithm
bnw_image = ch.green_to_bnw(img_denoised)


#TODO
# Perform DBscan


#TODO
# Iterate through bbox and center point list, call GET request on plantnet




#TODO
# Return results