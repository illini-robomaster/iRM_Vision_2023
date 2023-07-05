"""Hosts the fake camera class that reads from a file for testing."""
import sys
import time
import numpy as np
import cv2
import Utils

from Camera.camera_base import CameraBase


class simple_cv_camera(CameraBase):
    """Simple OpenCV camera."""

    # Needs to be calibrated per camera
    YAW_FOV_HALF = Utils.deg_to_rad(42) / 2
    PITCH_FOV_HALF = Utils.deg_to_rad(42) / 2

    def __init__(self, cfg):
        """Initialize a simple camera from defualt camera cam.

        Args:
            cfg (python object): shared config object
        """
        super().__init__(cfg)

        self.cap = cv2.VideoCapture(0)
        assert self.cap.isOpened()

    def get_frame(self):
        """Call to get a frame from the camera.

        Returns:
            np.ndarray: BGR image frame
        """
        ret, frame = self.cap.read()
        if not ret:
            raise
        return frame
