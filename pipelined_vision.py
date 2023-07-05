"""Deployment main file with pipelining."""

from Aiming.Aim import Aim
# from Detection.YOLO import Yolo
from Detection.two_stage_yolo import two_stage_yolo_detector
from Communication.communicator import UARTCommunicator
import config
import Utils
from functools import partial
from Launcher.pipeline_coordinator import pipeline_coordinator


def main():
    """Define the main while-true control loop that manages everything."""
    model = two_stage_yolo_detector(config, config.DEFAULT_ENEMY_TEAM)
    aimer = Aim(config)

    assert not config.DEBUG_DISPLAY, "display not supported in pipelined_vision.py"

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

    def tracker_and_viz(pred, enemy_team, stm32_state_dict):
        ret_dict = aimer.process_one(pred, enemy_team, stm32_state_dict)
        if ret_dict:
            communicator.process_one_packet(
                config.MOVE_YOKE, ret_dict['abs_yaw'], ret_dict['abs_pitch'])

    main_pipeline = pipeline_coordinator(stall_policy='debug_keep_all')

    main_pipeline.register_pipeline(
        stage=1,
        func=read_from_comm_and_camera,
        name='STM32 State Update',
        output_list=[
            'stm32_state_dict',
            'enemy_team',
            'raw_img_bgr'],
    )

    main_pipeline.register_pipeline(
        stage=2,
        func=partial(Utils.preprocess_img, cfg=config),
        name='Preprocessing',
        input_list=['raw_img_bgr'],
        output_list=['aug_img_dict'],
    )

    main_pipeline.register_pipeline(
        stage=3,
        func=model.detect,
        name='YOLO Detection',
        input_list=['aug_img_dict'],
        output_list=[
            'pred'],
    )

    main_pipeline.register_pipeline(
        stage=4,
        func=tracker_and_viz,
        name='Tracking and Visualization',
        input_list=[
            'pred',
            'enemy_team',
            'stm32_state_dict'],
    )

    main_pipeline.parse_all_stage()
    main_pipeline.start()


if __name__ == "__main__":
    main()
