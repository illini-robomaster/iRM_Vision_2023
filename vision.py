import os
import cv2
from Aiming.Aim import Aim
from Camera.camera import Camera
from Communication.communicator import Communicator
from Detection.darknet import Yolo
from variables import META_PATH, MODEL_CFG_PATH, WEIGHT_PATH


if __name__ == "__main__":
    camera = Camera()
    model = Yolo(MODEL_CFG_PATH, WEIGHT_PATH, META_PATH)
    aimer = Aim()
    communicator = Communicator()

    while True:
        frame = camera.get_frame()
        #!TODO: make this more elegant
        cv2.imwrite("tmp.jpg", frame)
        pred = model.detect("tmp.jpg")
        os.remove("tmp.jpg")
        # or should we use pid?
        yaw, pitch = aimer.get_rotation(pred)
        communicator.move(yaw, pitch)


