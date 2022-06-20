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
        
        ret_dict = aimer.process_one(pred, enemy_team, depth)

        show_frame = frame.copy()

        if ret_dict:
            packet = create_packet(config.MOVE_YOKE, pkt_seq, ret_dict['yaw_diff'], ret_dict['pitch_diff'])
            show_frame = cv2.circle(show_frame,
                                    (ret_dict['center_x'], ret_dict['center_y']), 10, (0, 255, 0), 10)
        else:
            show_frame = cv2.putText(show_frame, 'NOT FOUND', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            packet = create_packet(config.SEARCH_TARGET, pkt_seq, 0, 0)
        
        cv2.imshow('pred', show_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)

        if communicator is not None:
            communicator.write(packet)
        else:
            print("PACKET CREATED BUT SERIAL DEVICE IS NOT AVAILABLE!!!")

        pkt_seq += 1
