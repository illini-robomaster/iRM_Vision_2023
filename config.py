"""Config file that is shared across the whole project."""

# ========== Camera ==========
from Camera.read_from_file import fake_camera
from Camera.simple_cv import simple_cv_camera
# Define camera
# from Camera.d455 import D455_camera
# from Camera.mdvs import mdvs_camera
AUTOAIM_CAMERA = fake_camera

# This param needs to be tuned per arena / camera setup
EXPOSURE_TIME = 5

# Compute some constants and define camera to use
IMG_HEIGHT = 512
IMG_WIDTH = 640

IMG_CENTER_X = IMG_WIDTH // 2
IMG_CENTER_Y = IMG_HEIGHT // 2

ROTATE_180 = True  # Camera is mounted upside down

# ========== Trajectory Modeling ==========
GRAVITY_CONSTANT = 9.81         # acceleration due to gravity
INITIAL_BULLET_SPEED = 15.0     # empirically measured

# ========== Communication ==========
SEARCH_TARGET = b'ST'
MOVE_YOKE = b'MY'
PACK_END = b'ED'

# ========== DEBUGGING ==========

DEBUG_DISPLAY = True
DEBUG_PRINT = False
DEFAULT_ENEMY_TEAM = 'red'
