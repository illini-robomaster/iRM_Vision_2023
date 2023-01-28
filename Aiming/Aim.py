import numpy as np
import config
import Utils

from .Tracking import basic_tracker

class Aim:
    def __init__(self):
        self.tracker = basic_tracker()

    def process_one(self, pred_list, enemy_team, rgb_img, depth_map):
        assert enemy_team in ['blue', 'red']

        pred_list = self.tracker.fix_prediction(pred_list)

        self.tracker.register_one(pred_list, enemy_team, rgb_img, depth_map)

        closet_pred, closet_dist = self.get_closet_pred(pred_list, enemy_team, depth_map)
        if closet_pred is None:
            return None
        name, confidence, bbox = closet_pred
        center_x, center_y, width, height = bbox

        # Get yaw/pitch differences in radians
        yaw_diff, pitch_diff = self.get_rotation_angle(center_x, center_y)

        calibrated_yaw_diff, calibrated_pitch_diff = self.posterior_calibration(yaw_diff, pitch_diff, closet_dist)

        return {
            'yaw_diff': calibrated_yaw_diff,
            'pitch_diff': calibrated_pitch_diff,
            'center_x': center_x,
            'center_y': center_y,
        }
    
    def posterior_calibration(self, yaw_diff, pitch_diff, distance):
        """Given a set of naively estimated parameters, return calibrated parameters
        from a range table?

        Args:
            yaw_diff (float): yaw difference in radians
            pitch_diff (float): pitch difference in radians
            distance (float): distance to target in meters

        Returns:
            (float, float): calibrated yaw_diff, pitch_diff in radians
        """
        if distance >= 2**16:
            # In this case, distance is a rough estimation from bbox size
            return (yaw_diff, pitch_diff)
        else:
            # In this case, distance comes from D455 stereo estimation
            # TODO: compute a range table
            return (yaw_diff, pitch_diff)
    
    def get_closet_pred(self, pred_list, enemy_team, depth_map):
        closet_pred = None
        closet_dist = None # Cloest to camera in z-axis
        obj_of_interest = [f"armor_{enemy_team}"]
        for name, conf, bbox in pred_list:
            if name not in obj_of_interest: continue
            cur_dist = Utils.estimate_target_depth(bbox, depth_map)
            if closet_pred is None:
                closet_pred = (name, conf, bbox)
                closet_dist = cur_dist
            else:
                if cur_dist < closet_dist:
                    closet_pred = (name, conf, bbox)
                    closet_dist = cur_dist
        return closet_pred, closet_dist

    @staticmethod
    def get_rotation_angle(bbox_center_x, bbox_center_y):
        yaw_diff = (bbox_center_x - config.IMG_CENTER_X) * (config.RGBD_CAMERA.YAW_FOV_HALF / config.IMG_CENTER_X)
        pitch_diff = (bbox_center_y - config.IMG_CENTER_Y) * (config.RGBD_CAMERA.PITCH_FOV_HALF / config.IMG_CENTER_Y)

        return yaw_diff, pitch_diff
