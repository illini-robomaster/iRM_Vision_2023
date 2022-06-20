import time
import cv2
from Aiming.Aim import Aim
from Communication.communicator import serial_port, create_packet
from Detection.darknet import Yolo
import config

if __name__ == "__main__":
    model = Yolo(config.MODEL_CFG_PATH, config.WEIGHT_PATH, config.META_PATH)
    aimer = Aim()
    communicator = serial_port
    pkt_seq = 0

    rgbd_camera = config.RGBD_CAMERA(config.IMG_WIDTH, config.IMG_HEIGHT)

    while True:
        start = time.time()
        frame, depth = rgbd_camera.get_frame()

        pred = model.detect(frame)
        print('----------------\n',pred)
        elapsed = time.time()-start
        print('fps:',1./elapsed)

        # FIXME: GET ENEMY TEAM FROM THE REFEREE SYS
        enemy_team = 'blue'

        show_frame = frame.copy()
        cv2.imshow('pred', show_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
        
        ret = aimer.process_one(pred, enemy_team, depth)

        if ret:
            yaw_diff, pitch_diff = ret
            packet = create_packet(config.MOVE_YOKE, pkt_seq, yaw_diff, pitch_diff)
        else:
            packet = create_packet(config.SEARCH_TARGET, pkt_seq, 0, 0)

        if communicator is not None:
            communicator.write(packet)
        else:
            print("PACKET CREATED BUT SERIAL DEVICE IS NOT AVAILABLE!!!")

        pkt_seq += 1
