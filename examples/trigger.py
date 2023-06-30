import time
import threading
from Camera.mdvs import mdvs_camera
import config
from Communication.communicator import UARTCommunicator
import cv2
import numpy as np

last_time = 0
elapsed_time = 0
print_last_time = 0
fps_txt = 0

def trigger(camera_event, camera):
    while True:
        camera_event.wait()
        print("Triggering camera event!")
        # elapsed_time = time.time() - last_time
        # last_time = time.time()
        # fps = 1 / elapsed_time
        #
        # if time.time() - print_last_time > 0.1:
        #     fps_txt = fps
        #     print_last_time = time.time()
        # frame = cv2.putText(frame, "FPS: {:.1f}".format(fps_txt), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
        #                     cv2.LINE_AA)
        #
        # cv2.imshow("Press q to end", frame)
        # if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #     break
        camera_event.clear()

def main():
    camera_event = threading.Event()

    camera = mdvs_camera(config)

    trigger_thread = threading.Thread(target=trigger, args=(camera_event, camera))
    trigger_thread.start()

    camera.start_grabbing(camera_event)

    communicator = UARTCommunicator(config)
    if communicator.is_valid():
        print("OPENED SERIAL DEVICE AT: " + communicator.serial_port.name)
    else:
        print("SERIAL DEVICE IS NOT AVAILABLE!!!")
    communicator.start_listening()

    while True:
        time.sleep(1)  # main thread sleeps forever

if __name__ == "__main__":
    main()
