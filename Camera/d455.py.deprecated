import numpy as np
import pyrealsense2 as rs
import Utils

# TODO: make an abstract camera base class
class D455_camera(object):
    # Camera parameters from intelrealsense.com/depth-camera/d-455/
    # TODO: for better accuracy, use individual calibrated intrinsics
    YAW_FOV_HALF = Utils.deg_to_rad(87) / 2
    PITCH_FOV_HALF = Utils.deg_to_rad(58) / 2

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.pipeline = rs.pipeline()

        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)

        self.pipeline.start(self.config)

    def get_frame(self):
        RGBD_frame = self.pipeline.wait_for_frames()

        frame = np.array(RGBD_frame.get_color_frame().get_data()) # HW3
        depth = np.array(RGBD_frame.get_depth_frame().get_data()) # uint16 0-65535
        depth[depth == 65535] = 0 # mask ignored label to background

        return (frame, depth)
