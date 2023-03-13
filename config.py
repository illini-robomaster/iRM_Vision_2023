"""Config file that is shared across the whole project."""

# communication utils
from Camera.read_from_file import fake_camera
SEARCH_TARGET = b'ST'
MOVE_YOKE = b'MY'
PACK_END = b'ED'

# Define camera
# from Camera.d455 import D455_camera

AUTOAIM_CAMERA = fake_camera

# Compute some constants and define camera to use
IMG_HEIGHT = 360
IMG_WIDTH = 640

IMG_CENTER_X = IMG_WIDTH // 2
IMG_CENTER_Y = IMG_HEIGHT // 2

DEBUG_DISPLAY = False
DEFAULT_ENEMY_TEAM = 'red'
