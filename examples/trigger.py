import time
import threading
from Camera.mdvs import mdvs_camera
import config
from Communication.communicator import UARTCommunicator
import cv2
import numpy as np
import matplotlib.pyplot as plt


def trigger(camera_event, camera, communicator):
    last_time = 0
    print_last_time = 0
    fps_txt = 0
    fps_min = 200
    fps_max = 0
    fps_record = []
    stm32_txt = None
    trigger_count = 0
    while True:
        camera_event.wait()
        trigger_count += 1
        elapsed_time = time.time() - last_time
        last_time = time.time()
        fps = 1 / elapsed_time
        if 50 <= fps <= 150:
            fps_record.append(fps)
        if trigger_count >= 100 and fps < fps_min:
            fps_min = fps
        if trigger_count >= 100 and fps > fps_max:
            fps_max = fps

        if time.time() - print_last_time > 0.1:
            fps_txt = fps
            stm32_txt = communicator.stm32_state_dict
            print_last_time = time.time()

        frame = camera.get_frame_cache()
        frame = cv2.putText(frame, 'FPS: {:.1f} <= {:.1f} <= {:.1f}'.format(fps_min, fps_txt, fps_max), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 225, 0), 1, cv2.LINE_AA)
        if stm32_txt['my_color'] == 'red':
            my_color = (0, 0, 255)
        else:
            my_color = (255, 0, 0)
        frame = cv2.putText(frame, 'My Color: {}'.format(stm32_txt['my_color']), (25, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, my_color, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, 'Yaw: {:.1f}'.format(stm32_txt['cur_yaw'] * 180 / np.pi), (25, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        frame = cv2.putText(frame, 'Pitch: {:.1f}'.format(stm32_txt['cur_pitch'] * 180 / np.pi), (25, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        frame = cv2.putText(frame, 'Roll: {:.1f}'.format(stm32_txt['cur_roll'] * 180 / np.pi), (25, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        frame = cv2.putText(frame, 'Timestamp: {:.3f}'.format(communicator.stm32_state_dict['timestamp']), (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Press q to end', frame)
        camera_event.clear()
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            plt.hist(fps_record, bins=250)
            plt.ylabel('Number of Samples')
            plt.xlabel('FPS')
            plt.title('FPS Distribution Diagram')
            plt.show()
            break


def main():
    communicator = UARTCommunicator(config)
    if communicator.is_valid():
        print('OPENED SERIAL DEVICE AT: ' + communicator.serial_port.name)
    else:
        print('SERIAL DEVICE IS NOT AVAILABLE!!!')
    communicator.start_listening()

    camera_event = threading.Event()
    camera = mdvs_camera(config)

    trigger_thread = threading.Thread(target=trigger, args=(camera_event, camera, communicator))
    trigger_thread.start()

    camera.start_grabbing(camera_event)


if __name__ == '__main__':
    main()
