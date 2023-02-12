# darknet utils
DARKNET_LIB_PATH = r'/home/illinirm/darknet/libdarknet.so'
MODEL_CFG_PATH = r'/home/illinirm/darknet/dji_roco_preprocessed/yolov3-tiny-custom.cfg'
WEIGHT_PATH = r'/home/illinirm/darknet/dji_roco_preprocessed/yolov3-tiny-custom_final.weights'
META_PATH = r'/home/illinirm/darknet/dji_roco_preprocessed/roco.data'

# communication utils
SEARCH_TARGET = b'ST'
MOVE_YOKE = b'MY'
PACK_END = b'ED'

# from Camera.d455 import D455_camera
from Camera.read_from_file import fake_camera

AUTOAIM_CAMERA = fake_camera

# Compute some constants and define camera to use
IMG_HEIGHT=360
IMG_WIDTH=640

IMG_CENTER_X = IMG_WIDTH // 2
IMG_CENTER_Y = IMG_HEIGHT // 2

DEBUG_DISPLAY = True
