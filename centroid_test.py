from modules.cv_helpers import *
import cv2
import matplotlib.pyplot as plt
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN

image = cv2.imread('img/img1.png')

green_img = get_green(image)

arr = green_img
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        for k in range(arr.shape[2]):
            if arr[i, j, k] != 0:
                arr[i, j, k] = 255

denoised_image = cv2.fastNlMeansDenoisingColored(arr, None, h=100, templateWindowSize=7, searchWindowSize=21)
denoised_image = np.mean(denoised_image, axis=2)
denoised_image[denoised_image > 150] = 255
denoised_image[denoised_image < 150] = 0

x, y = binary_to_cartesian(denoised_image)

#DBSACN
white_points = np.transpose(np.array([x, y]))

# define the model
dbscan_model = DBSCAN(eps=5, min_samples=10)

# train the model
dbscan_model.fit(white_points)

# assign each data point to a cluster
dbscan_result = dbscan_model.fit_predict(white_points)

# get all of the unique clusters
dbscan_clusters = unique(dbscan_result)

# plot the DBSCAN clusters
x_cluster = []
y_cluster = []
for dbscan_cluster in dbscan_clusters:
    # get data points that fall in this cluster
    index = where(dbscan_result == dbscan_cluster)
    # make the plot
    x_cluster = white_points[index, 0]
    y_cluster = white_points[index, 1]

x_cluster = x_cluster[0]
y_cluster = y_cluster[0]

print(find_centroid_of_blob(x_cluster, y_cluster))