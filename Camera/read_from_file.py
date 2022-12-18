# Read from file
import sys
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

        assert len(sys.argv) == 2 # main py file; video file
        video_path = sys.argv[1]
        assert video_path.endswith('.mp4')

        self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened()

        self.alive = True # init to True

    def get_frame(self):
        if not self.cap.isOpened():
            self.alive = False
        
        ret, frame = self.cap.read()

        if not ret:
            self.alive = False
        
        if not self.alive:
            raise
        
        frame = cv2.resize(frame, (self.width, self.height))

        return (frame, np.ones_like(frame))
