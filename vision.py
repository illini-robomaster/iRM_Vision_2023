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
    if communicator.is_valid():
        print("OPENED SERIAL DEVICE AT: " + communicator.serial_port.name)
    else:
        print("SERIAL DEVICE IS NOT AVAILABLE!!!")
    communicator.start_listening()

    autoaim_camera = config.AUTOAIM_CAMERA(config)

    while True:
        start = time.time()
        stm32_state_dict = communicator.get_current_stm32_state()
        frame = autoaim_camera.get_frame()

        # TODO: add a global reset function if enemy functions change
        # (e.g., clear the buffer in the armor tracker)
        enemy_team = stm32_state_dict['enemy_color']
        model.change_color(enemy_team)

        pred = model.detect(frame)

        for i in range(len(pred)):
            name, conf, armor_type, bbox, armor = pred[i]
            # name from C++ string is in bytes; decoding is needed
            if isinstance(name, bytes):
                name_str = name.decode('utf-8')
            else:
                name_str = name
            pred[i] = (name_str, conf, armor_type, bbox, armor)

        elapsed = time.time() - start

        if config.DEBUG_DISPLAY:
            viz_frame = frame.copy()
            for _, _, _, bbox, _ in pred:
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
        ret_dict = aimer.process_one(pred, enemy_team, frame, stm32_state_dict)

        if config.DEBUG_DISPLAY:
            show_frame = frame.copy()

        if ret_dict:
            communicator.process_one_packet(
                config.MOVE_YOKE, ret_dict['abs_yaw'], ret_dict['abs_pitch'])
            if config.DEBUG_DISPLAY:
                # Reverse compute center_x and center_y from yaw angle
                yaw_diff = ret_dict['abs_yaw'] - stm32_state_dict['cur_yaw']
                pitch_diff = ret_dict['abs_pitch'] - stm32_state_dict['cur_pitch']
                target_x = -yaw_diff / (config.AUTOAIM_CAMERA.YAW_FOV_HALF /
                                        config.IMG_CENTER_X) + config.IMG_CENTER_X
                target_y = pitch_diff / (config.AUTOAIM_CAMERA.PITCH_FOV_HALF /
                                         config.IMG_CENTER_Y) + config.IMG_CENTER_Y
                show_frame = cv2.circle(show_frame,
                                        (int(target_x),
                                         int(target_y)),
                                        10, (0, 255, 0), 10)
                yaw_diff = ret_dict['uncalibrated_yaw'] - stm32_state_dict['cur_yaw']
                pitch_diff = ret_dict['uncalibrated_pitch'] - stm32_state_dict['cur_pitch']
                target_x = -yaw_diff / (config.AUTOAIM_CAMERA.YAW_FOV_HALF /
                                        config.IMG_CENTER_X) + config.IMG_CENTER_X
                target_y = pitch_diff / (config.AUTOAIM_CAMERA.PITCH_FOV_HALF /
                                         config.IMG_CENTER_Y) + config.IMG_CENTER_Y
                # import pdb; pdb.set_trace()
                show_frame = cv2.circle(show_frame,
                                        (int(target_x),
                                         int(target_y)),
                                        10, (0, 0, 255), 10)
        else:
            # communicator.process_one_packet(config.SEARCH_TARGET, 0, 0)
            if config.DEBUG_DISPLAY:
                show_frame = cv2.putText(show_frame, 'NOT FOUND', (50, 50),
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if config.DEBUG_PRINT:
            print('----------------\n', pred)
            print('fps:', 1. / elapsed)

        if config.DEBUG_DISPLAY:
            cv2.imshow('target', show_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)


if __name__ == "__main__":
    main()
