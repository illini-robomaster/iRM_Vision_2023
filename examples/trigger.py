import time

import cv2
from Camera.mdvs import mdvs_camera
import config


def main():
    camera = mdvs_camera(config)
    last_time = time.time()
    print_last_time = time.time()
    fps_text = 0

    while True:
        elasped_time = time.time() - last_time
        last_time = time.time()
        fps = 1.0 / elasped_time

        frame = camera.get_frame()

        if time.time() - print_last_time > 0.1:
            fps_text = fps
            # print("FPS: {:.3f}".format(fps))
            print_last_time = time.time()

        frame = cv2.putText(frame, "FPS: {:.1f}".format(fps_text), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()
