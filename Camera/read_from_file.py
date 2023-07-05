"""Hosts the fake camera class that reads from a file for testing."""
import sys
import time
import numpy as np
import cv2
import Utils

from Camera.camera_base import CameraBase


class fake_camera(CameraBase):
    """
    Fake camera class that mimics the behavior of the real camera.

    It reads from a video file instead.
    """

    # Needs to be calibrated per camera
    YAW_FOV_HALF = Utils.deg_to_rad(42) / 2
    PITCH_FOV_HALF = Utils.deg_to_rad(42) / 2

    def __init__(self, cfg):
        """Initialize fake camera.

        Args:
            cfg (python object): shared config object
        """
        super().__init__(cfg)

        assert len(sys.argv) == 2  # main py file; video file
        video_path = sys.argv[1]
        assert video_path.endswith('.mp4')

        self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened()

        self.alive = True  # init to True

        # Timing and frame counter are always in place for devlopment purpose
        self.timing = None
        self.frame_cnt = 0

    def get_frame(self):
        """Call to get a frame from the camera.

        Raises:
            Exception: raised when video file is exhausted

        Returns:
            np.ndarray: BGR image frame
        """
        if self.timing is None:
            self.timing = time.time()

        if not self.cap.isOpened():
            self.alive = False

        ret, frame = self.cap.read()

        if not ret:
            self.alive = False

        if not self.alive:
            print("Total frames: {}".format(self.frame_cnt))
            print("Total time: {}".format(time.time() - self.timing))
            print("FPS: {}".format(self.frame_cnt / (time.time() - self.timing)))
            raise Exception("Video file exhausted")

        if self.cfg.ROTATE_180:
            # Simulate upside down camera
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        frame = cv2.resize(frame, (self.cfg.IMG_WIDTH * 2, self.cfg.IMG_HEIGHT * 2), interpolation=cv2.INTER_LINEAR)

        self.frame_cnt += 1

        return frame
