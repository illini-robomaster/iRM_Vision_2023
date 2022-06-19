import numpy as np
import config

class Aim:
    def __init__(self):
        pass

    def process_one(self, pred_list, enemy_team, depth_map):
        assert enemy_team in ['blue', 'red']
        closet_pred = self.get_closet_pred(pred_list, enemy_team)
        if closet_pred is None:
            return None
        name, confidence, bbox = closet_pred
        center_x, center_y, width, height = bbox

        # Get yaw/pitch differences in radians
        yaw_diff, pitch_diff = self.get_rotation_angle(center_x, center_y)
        upper_left_x = int(center_x - width / 2)
        upper_left_y = int(center_y - height / 2)
        lower_right_x = int(center_x + width / 2)
        lower_right_y = int(center_y + height / 2)

        # Get distance to target
        depth_region = depth_map[upper_left_y:lower_right_y,upper_left_x:lower_right_x]
        estimated_depth = np.mean(depth_region[depth_region > 0])

        calibrated_yaw_diff, calibrated_pitch_diff = self.posterior_calibration(yaw_diff, pitch_diff, estimated_depth)

        return (calibrated_yaw_diff, calibrated_pitch_diff)
    
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
        # TODO: compute a range table
        return (yaw_diff, pitch_diff)
    
    def get_closet_pred(self, pred_list, enemy_team):
        closet_pred = None
        closet_l2_dist = None
        obj_of_interest = [f"armor_{enemy_team}"]
        for name, conf, bbox in pred_list:
            # name from C++ string is in bytes; decoding is needed
            if name.decode('utf-8') not in obj_of_interest: continue
            center_x, center_y, _, _ = bbox
            # TODO: use depth distance instead of L2 dist for watcher for easy targets?
            cur_l2_dist = np.square(config.IMG_CENTER_X // 2 - center_x) + np.square(config.IMG_CENTER_Y // 2 - center_y)
            if closet_pred is None:
                closet_pred = (name, conf, bbox)
                closet_l2_dist = cur_l2_dist
            else:
                if cur_l2_dist < closet_l2_dist:
                    closet_pred = (name, conf, bbox)
                    closet_l2_dist = cur_l2_dist
        return closet_pred

    def get_rotation_angle(bbox_center_x, bbox_center_y):
        yaw_diff = (bbox_center_x - config.IMG_CENTER_X) * (config.RGBD_CAMERA.YAW_FOV_HALF / config.IMG_CENTER_X)
        pitch_diff = (bbox_center_y - config.IMG_CENTER_Y) * (config.RGBD_CAMERA.PITCH_FOV_HALF / config.IMG_CENTER_Y)

        return yaw_diff, pitch_diff
