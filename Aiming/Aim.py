"""Hosts the Aim class, which is the main class for auto-aiming."""
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
        self.distance_estimator = pnp_estimator(self.CFG)

    def preprocess(self, pred_list, stm32_state_dict):
        """Preprocess predictions to compute armor distances and absolute yaw/pitch.

        Args:
            pred_list (list): a list of predictions
            stm32_state_dict (dict): a dictionary of stm32 state
            raw_rgb_image (np.ndarray): raw RGB image

        Returns:
            list: a list of tuples (armor_type, abs_yaw, abs_pitch, y_distance, z_distance)
        """
        # gimbal_yaw = stm32_state_dict['cur_yaw']
        # gimbal_pitch = stm32_state_dict['cur_pitch']

        #hardcoded gimbal's yaw and pitch to 0 to get relative angles
        gimbal_yaw = 0
        gimbal_pitch = 0

        camera_barrel_T = get_camera_barrel_T(self.CFG)

        ret_list = []

        for armor in pred_list:
            bbox = armor.bbox
            armor_type = armor.cls
            # armor_name, conf, armor_type, bbox, armor = pred
            armor_xyz, armor_yaw = self.distance_estimator.estimate_position(armor)
            armor_xyz = barrel_to_robot_T(gimbal_yaw, gimbal_pitch, armor_xyz)

            ret_list.append((armor_type, armor_xyz, bbox, armor_yaw))

        return ret_list

    def process_one(self, pred_list, enemy_team, stm32_state_dict):
        """Process one frame of predictions.

        Args:
            pred_list (list): a list of predictions
            enemy_team (str): 'blue' or 'red'
            rgb_img (np.ndarray): raw RGB image
            stm32_state_dict (dict): a dictionary of stm32 state

        Returns:
            dict: a dictionary of results
        """
        assert enemy_team in ['blue', 'red']

        # print('pred_list:',pred_list)
        # print('stm32 state: ',stm32_state_dict)

        observed_armors = self.preprocess(pred_list, stm32_state_dict)
        # print('observed armors:', observed_armors)
        if observed_armors == []:
            return None
        min_dist = 1000000
        for armor in observed_armors:
            target_pos = armor[1]
            target_center_x= armor[2][0]
            target_center_y = armor[2][1]
            target_dist = np.sqrt((target_center_x - self.CFG.IMG_CENTER_X)**2+(target_center_y - self.CFG.IMG_CENTER_Y)**2)
            #shoot the closest target
            if target_dist < min_dist:
                min_dist = target_dist
                _, target_pitch, target_yaw = Utils.cartesian_to_spherical(*target_pos)

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
