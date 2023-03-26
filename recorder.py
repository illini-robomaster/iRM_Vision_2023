"""
Main file for recording data.

PLease change the base_data_folder constant to a large disk
"""

import os
import time
import cv2
import config
import shutil
import datetime

TIME_PER_CLIP = 10  # in seconds
FPS_LIMIT = 1. / 100


def main():
    """Define the main while-true control loop that manages everything."""
    autoaim_camera = config.AUTOAIM_CAMERA(config)

    base_data_folder = "/tmp"
    cur_datetime = datetime.datetime.now()
    cur_data_folder = os.path.join(base_data_folder, cur_datetime.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(cur_data_folder, exist_ok=True)

    cur_video_writer = None
    cur_video_clip_idx = 0

    while True:
        if cur_video_writer is None:
            cur_free_space = shutil.disk_usage(base_data_folder).free  # in bytes
            cur_free_space = cur_free_space / 1024 / 1024 / 1024  # in GB
            if cur_free_space < 1:
                print("Not enough space left on disk. Idling...")
                time.sleep(100)
                continue
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_fn = os.path.join(cur_data_folder, "video_{}.mp4".format(cur_video_clip_idx))
            cur_video_writer = cv2.VideoWriter(
                video_fn, fourcc, 30, (config.IMG_WIDTH, config.IMG_HEIGHT))
            writer_start_time = time.time()
            last_frame_time = writer_start_time

        # Time right before capturing
        cur_time = time.time()
        # We don't want capture to be too frequent
        if cur_time - last_frame_time < FPS_LIMIT:
            continue
        frame = autoaim_camera.get_frame()
        last_frame_time = time.time()
        assert cur_video_writer is not None
        assert frame is not None
        cur_video_writer.write(frame)

        # Check if we need to start a new video clip
        # TIME_PER_CLIP seconds of data
        if last_frame_time - writer_start_time > TIME_PER_CLIP:
            cur_video_writer.release()
            cur_video_writer = None
            cur_video_clip_idx += 1


if __name__ == "__main__":
    main()
