"""Config file that is shared across the whole project."""
import os
import numpy as np

# ========== Camera ==========
from Camera.read_from_file import fake_camera
from Camera.simple_cv import simple_cv_camera
# Define camera
# from Camera.d455 import D455_camera
# from Camera.mdvs import mdvs_camera
AUTOAIM_CAMERA = fake_camera
# AUTOAIM_CAMERA = mdvs_camera
# This param needs to be tuned per arena / camera setup
EXPOSURE_TIME = 5

# Compute some constants and define camera to use
IMG_HEIGHT = 512
IMG_WIDTH = 640

IMG_CENTER_X = IMG_WIDTH // 2
IMG_CENTER_Y = IMG_HEIGHT // 2

ROTATE_180 = True  # Camera is mounted upside down

K = np.array([
    [776.10564907, 0, 314.26822299],
    [0, 775.89552525, 259.37110689],
    [0, 0, 1],
])

# 6mm mdvs lens Center cropping intrinsics
# K = np.array([
#     [1.52839e+03, 0, 286.925],
#     [0, 1527.11, 274.26],
#     [0, 0, 1],
# ])

# ========== Detection ==========
YOLO_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'shufflenet_no_nms.onnx')
INT8_QUANTIZATION = True
CALIB_IMG_DIR = 'calib_images'

# ========== Trajectory Modeling ==========
GRAVITY_CONSTANT = 9.81         # acceleration due to gravity
INITIAL_BULLET_SPEED = 10.0     # empirically measured

# ========== Communication ==========
SEARCH_TARGET = b'ST'
MOVE_YOKE = b'MY'
PACK_END = b'ED'

# ========== DEBUGGING ==========

DEBUG_DISPLAY = False
DEBUG_PRINT = False
DEFAULT_ENEMY_TEAM = 'red'
