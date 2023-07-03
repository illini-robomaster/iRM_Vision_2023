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

from Launcher.pipeline_coordinator import pipeline_coordinator

def main():
    """Define the main while-true control loop that manages everything."""
    model = two_stage_yolo_detector(config, config.DEFAULT_ENEMY_TEAM)
    aimer = Aim(config)

    assert config.DEBUG_DISPLAY == False, "display not supported in pipelined_vision.py"

    communicator = UARTCommunicator(config)
    communicator.start_listening()

    autoaim_camera = config.AUTOAIM_CAMERA(config)

    if communicator.is_valid():
        print("OPENED SERIAL DEVICE AT: " + communicator.serial_port.name)
    else:
        print("SERIAL DEVICE IS NOT AVAILABLE!!!")
    
    def read_from_comm_and_camera():
        stm32_state_dict = communicator.get_current_stm32_state()
        enemy_team = stm32_state_dict['enemy_color']
        model.change_target_color(enemy_team)
        frame = autoaim_camera.get_frame()
        return stm32_state_dict, enemy_team, frame
    
    def tracker_and_viz(pred, enemy_team, resized_bgr_frame, stm32_state_dict):
        ret_dict = aimer.process_one(pred, enemy_team, resized_bgr_frame, stm32_state_dict)

        if config.DEBUG_DISPLAY:
            show_frame = resized_bgr_frame.copy()

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
    
    main_pipeline = pipeline_coordinator(stall_policy='keep_all')

    main_pipeline.register_pipeline(stage=1,
                                    func=read_from_comm_and_camera,
                                    name='STM32 State Update',
                                    output_list=['stm32_state_dict', 'enemy_team', 'resized_bgr_frame'],
                                    )
    
    main_pipeline.register_pipeline(stage=2,
                                    func=model.detect,
                                    name='YOLO Detection',
                                    input_list=['resized_bgr_frame'],
                                    output_list=['pred'],
                                    )

    main_pipeline.register_pipeline(stage=3,
                                    func=tracker_and_viz,
                                    name='Tracking and Visualization',
                                    input_list=['pred', 'enemy_team', 'resized_bgr_frame', 'stm32_state_dict'],
                                    )
    
    main_pipeline.parse_all_stage()
    main_pipeline.start()

if __name__ == "__main__":
    main()
