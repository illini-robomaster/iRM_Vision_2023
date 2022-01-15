import os, time
import cv2
from Aiming.Aim import Aim
from Camera.camera import Camera
from Communication.communicator import Communicator
from Detection.darknet import Yolo
from variables import *

if __name__ == "__main__":
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    model = Yolo(MODEL_CFG_PATH, WEIGHT_PATH, META_PATH)
    aimer = Aim()
    communicator = Communicator()
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

            # TODO: or should we use pid?
            #yaw, pitch = aimer.get_rotation(pred)
            #communicator.move(yaw, pitch)
    else:
        print('Failed to open camera!')


