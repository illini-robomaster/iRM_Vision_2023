# Read from file
import sys
import time
import numpy as np
import cv2
import Utils

# TODO: make an abstract camera base class


class fake_camera(object):
    # Needs to be calibrated per camera
    YAW_FOV_HALF = Utils.deg_to_rad(42) / 2
    PITCH_FOV_HALF = Utils.deg_to_rad(42) / 2

    def __init__(self, width, height):
        self.width = width
        self.height = height

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

        frame = cv2.resize(frame, (self.width, self.height))

        self.frame_cnt += 1

        return frame
