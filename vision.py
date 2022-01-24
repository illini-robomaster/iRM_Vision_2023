import time
import cv2
from Aiming.Aim import Aim
from Communication.communicator import Communicator, create_packet, send_packet, serial_port
from Detection.darknet import Yolo
from variables import *

if __name__ == "__main__":
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    model = Yolo(MODEL_CFG_PATH, WEIGHT_PATH, META_PATH)
    aimer = Aim()
    communicator = serial_port
    pkt_seq = 0

    if cap.isOpened():
        while True:
            start = time.time()
            ret, frame = cap.read()
            if do_display:
                cv2.imshow('CSI Camera',frame)
                # This also acts as
                keyCode = cv2.waitKey(30) & 0xff
                # Stop the program on the ESC key
                if keyCode == 27:
                   break

            pred = model.detect(frame)
            print('----------------\n',pred)
            elapsed = time.time()-start
            print('fps:',1./elapsed)

            # if len(pred) == 0:
            #     packet = create_packet(SEARCH_TARGET, b"", pkt_seq)
            # else:
            #     hori_offset, vert_offset = aimer.get_rotation(pred)
            #     hori_offset = hori_offset.to_bytes(4,'big')
            #     vert_offset = vert_offset.to_bytes(4,'big')
            #     packet = create_packet(MOVE_YOKE, hori_offset + vert_offset, pkt_seq)
            # send_packet(communicator, packet)

            # pkt_seq += 1

    else:
        print('Failed to open camera!')


