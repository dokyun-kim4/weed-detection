import pyrealsense2 as rs
import numpy as np
import cv2

def initiate_camera_connection():
    """
    Connect to the realsense camera pipeline, set the configuration of the
    camera, and then return the pipeline object.

    RETURNS
    -------
        Realsense Camera Pipeline Object
    """
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start()

    return pipeline

def get_opencv_arr_from_frame(pipeline):
    """
    Given the realsense camera pipeline, return two arrays: rgb and depth in
    numpy arr.

    PARAMETERS
    ----------
        pipeline: pipeline obj

    RETURNS
    -------
        (rbg: np array, depth: np array)
    """
    frame = pipeline.wait_for_frames()
    color_frame = frame.get_color_frame()
    depth_frame = frame.get_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    return color_image, depth_image

def pixels_to_meters(pipeline, x, y):
    """
    Given x, y coords in pixels, return their distance in the real world in 
    meters.

    PARAMETERS
    ----------
        x: int
        y: int

    RETURNS
    -------
        (x, y) -> in meters
    """
    frame = pipeline.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    depth = depth_frame.get_distance(x, y)
    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)

    return (depth_point[0], depth_point[1])