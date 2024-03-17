import pyrealsense2 as rs
import numpy as np
import cv2
import modules.realsense_helpers as rsh

pipeline = rsh.initiate_camera_connection()

while True:
    x = 300
    y = 240
    color_image, depth_image = rsh.get_opencv_arr_from_frame(pipeline)
    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
    depth_cm = cv2.circle(depth_cm, (x, y), radius=5, color=(0, 255, 255), thickness=-1)

    depth_point = rsh.pixels_to_meters(pipeline, x, y)
    print(depth_point)

    cv2.imshow('rgb', depth_cm)

    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()