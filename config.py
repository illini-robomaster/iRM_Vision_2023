"""Config file that is shared across the whole project."""

# communication utils
from Camera.simple_cv import simple_cv_camera
SEARCH_TARGET = b'ST'
MOVE_YOKE = b'MY'
PACK_END = b'ED'

# Define camera
# from Camera.d455 import D455_camera
# from Camera.mdvs import mdvs_camera
from Camera.read_from_file import fake_camera

AUTOAIM_CAMERA = fake_camera

# This param needs to be tuned per arena / camera setup
EXPOSURE_TIME = 5

# Compute some constants and define camera to use
IMG_HEIGHT = 512
IMG_WIDTH = 640

IMG_CENTER_X = IMG_WIDTH // 2
IMG_CENTER_Y = IMG_HEIGHT // 2

DEBUG_DISPLAY = False
DEBUG_PRINT = True
DEFAULT_ENEMY_TEAM = 'red'

ROTATE_180 = True  # Camera is mounted upside down
