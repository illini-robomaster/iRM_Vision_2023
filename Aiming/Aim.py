"""Hosts the Aim class, which is the main class for auto-aiming."""
from .Tracking import basic_tracker
from .DistEst import pnp_estimator
from .TrajModel import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import Utils

from IPython import embed


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
        self.tracker = basic_tracker(self.CFG)
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

        # TODO(roger): compute actual transformation
        camera_barrel_T = np.eye(4)

        r = R.from_euler('zyx', [0, gimbal_yaw, gimbal_pitch], degrees=False)
        gimbal_T = np.eye(4)
        gimbal_T[:3, :3] = r.as_matrix()

        ret_list = []

        for pred in pred_list:
            armor_name, conf, armor_type, bbox, armor = pred
            armor_xyz, armor_yaw = self.distance_estimator.estimate_position(armor, rgb_img)

            tmp_armor_pose = np.eye(4)
            tmp_armor_pose[:3, 3:] = armor_xyz

            # Use barrel front as the origin
            tmp_armor_pose = camera_barrel_T @ tmp_armor_pose
            # Use gimbal and initialized yaw as the origin
            tmp_armor_pose = gimbal_T @ tmp_armor_pose

            ret_list.append((armor_type, tmp_armor_pose[:3, 3], armor_yaw))

        return ret_list

    def pick_target(self, distance_angle_list, stm32_state_dict):
        """Select a priortized target from a list of targets.

        Args:
            distance_angle_list (list): list of targets returned from tracker
            stm32_state_dict (dict): a dictionary of stm32 state

        Returns:
            target: a tuple of (target_pitch, target_yaw, target_y_dist, target_z_dist)
        """
        gimbal_pitch = stm32_state_dict['cur_pitch']
        gimbal_yaw = stm32_state_dict['cur_yaw']

        target_pitch = None
        target_yaw = None
        target_dist = None
        min_angle_diff = 9999

        for dist, predicted_pitch, predicted_yaw in distance_angle_list:
            pitch_diff = Utils.get_radian_diff(predicted_pitch, gimbal_pitch)
            yaw_diff = Utils.get_radian_diff(predicted_yaw, gimbal_yaw)
            angle_diff = pitch_diff + yaw_diff
            if angle_diff < min_angle_diff:
                target_pitch = predicted_pitch
                target_yaw = predicted_yaw
                target_dist = dist
                min_angle_diff = angle_diff

        return target_pitch, target_yaw, target_dist

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

        distance_angle_list, final_id_list = self.tracker.process_one(
            observed_armors, enemy_team, rgb_img)

        # TODO: integrate this into tracking for consistent tracking
        target_dist_angle_tuple = self.pick_target(distance_angle_list, stm32_state_dict)

        if target_dist_angle_tuple[0] is None:
            return None

        target_pitch, target_yaw, target_dist = target_dist_angle_tuple

        calibrated_pitch, calibrated_yaw = self.posterior_calibration(
            target_pitch, target_yaw, target_dist)

        return {
            # 'yaw_diff': calibrated_yaw_diff,
            # 'pitch_diff': calibrated_pitch_diff,
            'abs_yaw': calibrated_yaw,
            'abs_pitch': calibrated_pitch,
            # 'center_x': center_x,
            # 'center_y': center_y,
            # 'final_bbox_list': final_bbox_list,
            # 'final_id_list': final_id_list,
        }

    def posterior_calibration(self, raw_pitch, raw_yaw, target_dist):
        """Given a set of naively estimated parameters, return calibrated parameters.

        Idea:
            Use a range table?

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

    # def get_rotation_angle(self, bbox_center_x, bbox_center_y):
    #     """Given a bounding box center, return the yaw/pitch difference in radians.

    #     Args:
    #         bbox_center_x (float): x coordinate of the center of the bounding box
    #         bbox_center_y (float): y coordinate of the center of the bounding box

    #     Returns:
    #         (float, float): yaw_diff, pitch_diff in radians
    #     """
    #     yaw_diff = (bbox_center_x - self.CFG.IMG_CENTER_X) * \
    #         (self.CFG.AUTOAIM_CAMERA.YAW_FOV_HALF / self.CFG.IMG_CENTER_X)
    #     pitch_diff = (bbox_center_y - self.CFG.IMG_CENTER_Y) * \
    #         (self.CFG.AUTOAIM_CAMERA.PITCH_FOV_HALF / self.CFG.IMG_CENTER_Y)

    #     yaw_diff = -yaw_diff  # counter-clockwise is positive
    #     pitch_diff = pitch_diff  # down is positive

    #     return yaw_diff, pitch_diff
