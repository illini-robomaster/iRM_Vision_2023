"""
Main file for both development and deployment.

For development purpose, you may want to turn on the DEBUG_DISPLAY flag
in config.py
"""

import time
import cv2
from Aiming.Aim import Aim
# from Detection.YOLO import Yolo
from Detection.two_stage_yolo import two_stage_yolo_detector
from Communication.communicator import UARTCommunicator
import config
import Utils


def main():
    """Define the main while-true control loop that manages everything."""
    model = two_stage_yolo_detector(config, config.DEFAULT_ENEMY_TEAM)
    # model = Yolo(config.MODEL_CFG_PATH, config.WEIGHT_PATH, config.META_PATH)
    aimer = Aim(config)

    communicator = UARTCommunicator(config)
    communicator.start_listening()

    autoaim_camera = config.AUTOAIM_CAMERA(config)

    if communicator.is_valid():
        print("OPENED SERIAL DEVICE AT: " + communicator.serial_port.name)
    else:
        print("SERIAL DEVICE IS NOT AVAILABLE!!!")

    while True:
        stm32_state_dict = communicator.get_current_stm32_state()
        print(stm32_state_dict)
        frame = autoaim_camera.get_frame()
        tmp_dict = Utils.preprocess_img_step1(frame, config)
        aug_img_dict = Utils.preprocess_img_step2(tmp_dict, config)

        # TODO: add a global reset function if enemy functions change
        # (e.g., clear the buffer in the armor tracker)
        enemy_team = stm32_state_dict['enemy_color']
        model.change_target_color(enemy_team)

        pred = model.detect(aug_img_dict)

        # Tracking and filtering
        # Pour all predictions into the aimer, which returns relative angles
        ret_dict = aimer.process_one(pred, enemy_team, stm32_state_dict)

        if config.DEBUG_DISPLAY:
            show_frame = aug_img_dict['resized_img_bgr'].copy()

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

        if config.DEBUG_DISPLAY:
            cv2.imshow('target', show_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)


if __name__ == "__main__":
    main()
