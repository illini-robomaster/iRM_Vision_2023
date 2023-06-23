"""Hosts the Aim class, which is the main class for auto-aiming."""
from .Tracking import tracker
from .DistEst import pnp_estimator
from .TrajModel import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import Utils


class Aim:
    """
    The main class for auto-aiming.

    Its workflow:
        1. takes in a list of predictions from the detection module
        2. apply tracker to fill in missing predictions / append or associate predictions
        3. It selects a single armor to hit
        4. It computes the yaw/pitch difference to the armor
        5. Finally, apply posterior calibration (e.g., range table to hit far targets)
    """

    def __init__(self, config):
        """Get the config and initialize the tracker.

        Args:
            config (python object): shared config
        """
        self.CFG = config
        self.tracker = tracker(self.CFG)
        self.distance_estimator = pnp_estimator(self.CFG)

    def preprocess(self, pred_list, stm32_state_dict, rgb_img):
        """Preprocess predictions to compute armor distances and absolute yaw/pitch.

        Args:
            pred_list (list): a list of predictions
            stm32_state_dict (dict): a dictionary of stm32 state

        Returns:
            list: a list of tuples (armor_type, abs_yaw, abs_pitch, y_distance, z_distance)
        """
        gimbal_yaw = stm32_state_dict['cur_yaw']
        gimbal_pitch = stm32_state_dict['cur_pitch']

        # TODO(roger): support only steering robot now
        x_offset = 0.1
        y_offset = 0.05
        z_offset = 0.085
        camera_barrel_T = np.eye(4)

        # TODO(roger): set a constant and test transform sign
        gimbal_yaw = 0
        gimbal_pitch = 0

        r = R.from_euler('zyx', [0, gimbal_pitch, gimbal_yaw], degrees=False)
        gimbal_T = np.eye(4)
        gimbal_T[:3, :3] = r.as_matrix()

        ret_list = []

        for pred in pred_list:
            armor_name, conf, armor_type, bbox, armor = pred
            armor_xyz, armor_yaw = self.distance_estimator.estimate_position(armor)

            tmp_armor_pose = np.eye(4)
            tmp_armor_pose[:3, 3:] = armor_xyz

            # Use barrel front as the origin
            tmp_armor_pose = camera_barrel_T @ tmp_armor_pose
            # Use gimbal and initialized yaw as the origin
            tmp_armor_pose = gimbal_T @ tmp_armor_pose

            ret_list.append((armor_type, tmp_armor_pose[:3, 3], bbox, armor_yaw))

        return ret_list

    def process_one(self, pred_list, enemy_team, rgb_img, stm32_state_dict):
        """Process one frame of predictions.

        Args:
            pred_list (list): a list of predictions
            enemy_team (str): 'blue' or 'red'
            rgb_img (np.ndarray): RGB image
            stm32_state_dict (dict): a dictionary of stm32 state

        Returns:
            dict: a dictionary of results
        """
        assert enemy_team in ['blue', 'red']

        observed_armors = self.preprocess(pred_list, stm32_state_dict, rgb_img)

        target_dist_angle_tuple = self.tracker.process_one(observed_armors, stm32_state_dict)

        if target_dist_angle_tuple is None or target_dist_angle_tuple[0] is None:
            return None

        target_dist, target_pitch, target_yaw = target_dist_angle_tuple

        calibrated_pitch, calibrated_yaw = self.posterior_calibration(
            target_pitch, target_yaw, target_dist)

        return {
            'abs_yaw': calibrated_yaw,
            'abs_pitch': calibrated_pitch,
            'uncalibrated_yaw': target_yaw,
            'uncalibrated_pitch': target_pitch,
            'target_dist': target_dist,
        }

    def posterior_calibration(self, raw_pitch, raw_yaw, target_dist):
        """Given a set of naively estimated parameters, return calibrated parameters.

        Args:
            yaw_diff (float): yaw difference in radians
            pitch_diff (float): pitch difference in radians
            distance (float): distance to target in meters

        Returns:
            (float, float): calibrated yaw_diff, pitch_diff in radians
        """
        target_pitch = raw_pitch
        target_yaw = raw_yaw

        # Gravity calibration
        pitch_diff = calibrate_pitch_gravity(self.CFG, target_dist)
        target_pitch -= pitch_diff

        return (target_pitch, target_yaw)
