# darknet utils
DARKNET_LIB_PATH = r'/home/illinirm/darknet/libdarknet.so'
MODEL_CFG_PATH = r'/home/illinirm/darknet/dji_roco_preprocessed/yolov3-tiny-custom.cfg'
WEIGHT_PATH = r'/home/illinirm/darknet/dji_roco_preprocessed/yolov3-tiny-custom_final.weights'
META_PATH = r'/home/illinirm/darknet/dji_roco_preprocessed/roco.data'
TMP_IMG = r'/dev/shm/tmp.jpg'

# communication utils
SERIAL_PORT = "/dev/ttyTHS1"
SEARCH_TARGET = b'\x04\x02'
MOVE_YOKE = b'\x04\x03'

max_frame = 20
gst_pipeline = f'nvarguscamerasrc sensor_mode=0 ! video/x-raw(memory:NVMM), width=(int)416, height=(int)416,format=(string)NV12, framerate=(fraction){max_frame}/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! videobalance contrast=1 brightness=-0.2  ! appsink max-buffers=1 drop=true'
do_display = False

