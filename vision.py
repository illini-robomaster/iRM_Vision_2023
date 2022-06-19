import time
import cv2
from Aiming.Aim import Aim
# from Communication.communicator import serial_port
from Detection.darknet import Yolo
import config
import numpy as np

def pad_packet(header, seq_num, yaw_offset, pitch_offset):
    assert header in [b'ST', b'MY']
    packet = header
    assert seq_num >= 0 and seq_num < 2**32 - 1 # uint32
    packet += seq_num.to_bytes(4, 'big')
    # YAW/PITCH offset should not be too high
    assert yaw_offset > -config.RGBD_CAMERA.YAW_FOV_HALF and yaw_offset < config.RGBD_CAMERA.YAW_FOV_HALF
    assert pitch_offset > -config.RGBD_CAMERA.PITCH_FOV_HALF and pitch_offset < config.RGBD_CAMERA.PITCH_FOV_HALF
    discrete_yaw_offset = int(yaw_offset * 100000)
    discrete_pitch_offset = int(pitch_offset * 100000)
    packet += (discrete_yaw_offset & 0xFFFFFFFF).to_bytes(4, 'big')
    packet += (discrete_pitch_offset & 0xFFFFFFFF).to_bytes(4, 'big')
    # ENDING
    packet += b'ED'
    return packet

if __name__ == "__main__":
    model = Yolo(config.MODEL_CFG_PATH, config.WEIGHT_PATH, config.META_PATH)
    aimer = Aim()
    # communicator = serial_port
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
        
        yaw_diff, pitch_diff = aimer.process_one(pred, enemy_team, depth)

        if pred:
            packet = pad_packet(b'MY', pkt_seq, yaw_diff, pitch_diff)
        else:
            packet = pad_packet(b'ST', pkt_seq, 0, 0)
        #send_packet(communicator, packet)

        pkt_seq += 1
