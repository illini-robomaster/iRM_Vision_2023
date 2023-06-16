"""Implemented EKF tracker with filterpy library."""
import numpy as np
from scipy.spatial.transform import Rotation as R

import Utils

# TODO: move this to config
FRAME_BUFFER_SIZE = 10


class tracked_armor(object):
    """A class that represents a tracked armor.

    It stores the history of bounding boxes and ROIs, and can predict the
    bounding box of the next frame.
    """

    def __init__(self, armor_type, armor_xyz, armor_yaw, frame_tick):
        """Initialize from prediction.

        Args:
            bbox (tuple): (center_x, center_y, w, h)
            roi (np.ndarray): ROI of the armor
            frame_tick (int): frame tick
            armor_id (int): unique ID
        """
        self.armor_type = armor_type
        self.armor_xyz_buffer = [armor_xyz]
        self.armor_yaw_buffer = [armor_yaw]
        self.observed_frame_tick = [frame_tick]
        self.armor_id = -1  # unique ID

    def compute_cost(self, other_armor):
        """Compute the cost of matching this armor with another armor.

        Args:
            other_armor (tracked_armor): another armor

        Returns:
            float: cost
        """
        assert isinstance(other_armor, tracked_armor)
        if self.armor_type != other_armor.armor_type:
            return 99999999
        # TODO: use more sophisticated metrics (e.g., RGB) as cost function
        my_xyz = self.armor_xyz_buffer[-1]
        other_xyz = other_armor.armor_xyz_buffer[-1]
        return np.sqrt(np.sum(np.square(my_xyz - other_xyz)))

    def update(self, other_armor, frame_tick):
        """Update the state of this armor with matched armor.

        Args:
            other_armor (tracked_armor): another armor
            frame_tick (int): frame tick
        """
        # Only call if these two armors are matched
        assert len(other_armor.armor_xyz_buffer) == 1
        self.armor_xyz_buffer.append(other_armor.armor_xyz_buffer[-1])
        self.armor_yaw_buffer.append(other_armor.armor_yaw_buffer[-1])
        self.observed_frame_tick.append(frame_tick)

        # Maintain each armor's buffer so that anything older than
        # FRAME_BUFFER_SIZE is dropped
        self.armor_xyz_buffer = self.armor_xyz_buffer[-FRAME_BUFFER_SIZE:]
        self.armor_yaw_buffer = self.armor_yaw_buffer[-FRAME_BUFFER_SIZE:]
        self.observed_frame_tick = self.observed_frame_tick[-FRAME_BUFFER_SIZE:]

    def predict_distance_angle(self, cur_frame_tick):
        """Predict the distance and angle of the tracked armor at cur frame tick.

        Args:
            cur_frame_tick (int): current frame tick

        Returns:
            tuple: (y_distance, z_distance, yaw, pitch)
        """
        # TODO(roger): predict motion
        xyz_pos = self.armor_xyz_buffer[-1]

        dist = np.sqrt(np.sum(np.square(xyz_pos)))

        gimbal_pitch = np.arctan2(xyz_pos[1], xyz_pos[2])
        gimbal_yaw = -np.arctan2(xyz_pos[0], xyz_pos[2])

        return dist, gimbal_pitch, gimbal_yaw
