"""
Main file for both development and deployment.

For development purpose, you may want to turn on the DEBUG_DISPLAY flag
in config.py
"""

import time
import cv2
from Aiming.Aim import Aim
# from Detection.YOLO import Yolo
from Detection.CV_mix_DL import cv_mix_dl_detector
from Communication.communicator import UARTCommunicator
import config


def main():
    """Define the main while-true control loop that manages everything."""
    model = cv_mix_dl_detector(config, config.DEFAULT_ENEMY_TEAM)
    # model = Yolo(config.MODEL_CFG_PATH, config.WEIGHT_PATH, config.META_PATH)
    aimer = Aim(config)

    communicator = UARTCommunicator(config)

    autoaim_camera = config.AUTOAIM_CAMERA(config)

    if communicator.is_valid():
        print("OPENED SERIAL DEVICE AT: " + communicator.serial_port.name)
    else:
        print("SERIAL DEVICE IS NOT AVAILABLE!!!")

    while True:
        start = time.time()
        frame = autoaim_camera.get_frame()

        # TODO: add a global reset function if enemy functions change
        # (e.g., clear the buffer in the armor tracker)
        stm32_state_dict = communicator.get_current_stm32_state()
        enemy_team = stm32_state_dict['enemy_color']
        model.change_color(enemy_team)

        pred = model.detect(frame)

        for i in range(len(pred)):
            name, conf, bbox = pred[i]
            # name from C++ string is in bytes; decoding is needed
            if isinstance(name, bytes):
                name_str = name.decode('utf-8')
            else:
                name_str = name
            pred[i] = (name_str, conf, bbox)

        elapsed = time.time() - start

        if config.DEBUG_DISPLAY:
            viz_frame = frame.copy()
            for _, _, bbox in pred:
                lower_x = int(bbox[0] - bbox[2] / 2)
                lower_y = int(bbox[1] - bbox[3] / 2)
                upper_x = int(bbox[0] + bbox[2] / 2)
                upper_y = int(bbox[1] + bbox[3] / 2)
                viz_frame = cv2.rectangle(
                    viz_frame, (lower_x, lower_y), (upper_x, upper_y), (0, 255, 0), 2)
            cv2.imshow('all_detected', viz_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)

        # Tracking and filtering
        # Pour all predictions into the aimer, which returns relative angles
        ret_dict = aimer.process_one(pred, enemy_team, frame)

        if config.DEBUG_DISPLAY:
            viz_frame = frame.copy()
            if ret_dict:
                for i in range(len(ret_dict['final_bbox_list'])):
                    bbox = ret_dict['final_bbox_list'][i]
                    unique_id = ret_dict['final_id_list'][i]
                    lower_x = int(bbox[0] - bbox[2] / 2)
                    lower_y = int(bbox[1] - bbox[3] / 2)
                    upper_x = int(bbox[0] + bbox[2] / 2)
                    upper_y = int(bbox[1] + bbox[3] / 2)
                    viz_frame = cv2.rectangle(
                        viz_frame, (lower_x, lower_y), (upper_x, upper_y), (0, 255, 0), 2)
                    viz_frame = cv2.putText(viz_frame, str(unique_id), (lower_x, lower_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('filtered_detected', viz_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)

        # TODO: put this into debug display
        show_frame = frame.copy()

        if ret_dict:
            communicator.process_one_packet(
                config.MOVE_YOKE, ret_dict['yaw_diff'], ret_dict['pitch_diff'])
            show_frame = cv2.circle(show_frame,
                                    (int(ret_dict['center_x']),
                                     int(ret_dict['center_y'])),
                                    10, (0, 255, 0), 10)
        else:
            show_frame = cv2.putText(show_frame, 'NOT FOUND', (50, 50),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            communicator.process_one_packet(config.SEARCH_TARGET, 0, 0)

        if config.DEBUG_PRINT:
            print('----------------\n', pred)
            print('fps:', 1. / elapsed)

        if config.DEBUG_DISPLAY:
            cv2.imshow('target', show_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)


if __name__ == "__main__":
    main()
