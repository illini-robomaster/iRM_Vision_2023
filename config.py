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
PACK_START = b'ST'
PACK_END = b'ED'

GIMBAL_CMD_ID = 0x00
COLOR_CMD_ID = 0x01
CHASSIS_CMD_ID = 0x02
SELFCHECK_CMD_ID = 0x03
ARM_CMD_ID = 0x04

# mapping from cmd_id to data section length of the packet, unit: byte
# packet length = data length + 9
CMD_TO_LEN = {
    GIMBAL_CMD_ID: 10,
    COLOR_CMD_ID: 1,
    CHASSIS_CMD_ID: 12,
    SELFCHECK_CMD_ID: 2,
    ARM_CMD_ID: 24,
}
# length of Header + Tail = 9 bytes
HT_LEN = 9

# 0 for search target
# 1 for move yoke
GIMBAL_MODE = [
    'ST',
    'MY',
]

# 0 for echo
# 1 for ignore
SELFCHECK_MODE = [
    'FLUSH',
    'ECHO',
    'ID',
]

SEQNUM_OFFSET = 2
DATA_LENGTH_OFFSET = SEQNUM_OFFSET + 2
CMD_ID_OFFSET = DATA_LENGTH_OFFSET + 1
DATA_OFFSET = CMD_ID_OFFSET + 1

# ========== DEBUGGING ==========

DEBUG_DISPLAY = True
DEBUG_PRINT = False
DEFAULT_ENEMY_TEAM = 'red'
