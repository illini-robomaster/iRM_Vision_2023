"""Hosts the Aim class, which is the main class for auto-aiming."""
# from .Tracking import basic_tracker
from .Tracking import KF_tracker, KalmanTracker


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
        self.tracker = KF_tracker(self.CFG)

    def process_one(self, pred_list, enemy_team, rgb_img, tracker):
        """Process one frame of predictions.

        Args:
            pred_list (list): a list of predictions
            enemy_team (str): 'blue' or 'red'
            rgb_img (np.ndarray): RGB image

        Returns:
            dict: a dictionary of results
        """
        assert enemy_team in ['blue', 'red']

        # TODO: use assertion to check enemy_team

        final_bbox_list, final_id_list = self.tracker.process_one(
            pred_list, enemy_team, rgb_img, tracker)

        # TODO: integrate this into tracking for consistent tracking
        closet_pred, closet_dist = self.get_closet_pred(
            final_bbox_list, rgb_img)

        if closet_pred is None:
            return None
        center_x, center_y, width, height = closet_pred

        # Get yaw/pitch differences in radians
        yaw_diff, pitch_diff = self.get_rotation_angle(center_x, center_y)

        calibrated_yaw_diff, calibrated_pitch_diff = self.posterior_calibration(
            yaw_diff, pitch_diff, closet_dist)

        return {
            'yaw_diff': calibrated_yaw_diff,
            'pitch_diff': calibrated_pitch_diff,
            'center_x': center_x,
            'center_y': center_y,
            'final_bbox_list': final_bbox_list,
            'final_id_list': final_id_list,
        }

    def posterior_calibration(self, yaw_diff, pitch_diff, distance):
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
        if distance >= 2**16:
            # In this case, distance is a rough estimation from bbox size
            return (yaw_diff, pitch_diff)
        else:
            # In this case, distance comes from D455 stereo estimation
            # TODO: compute a range table
            return (yaw_diff, pitch_diff)

    def get_closet_pred(self, bbox_list, rgb_img):
        """Get the closet prediction to camera focal point.

        Args:
            bbox_list (list): a list of bounding boxes
            rgb_img (np.ndarray): RGB image

        Returns:
            (bbox_list, float): closet_pred, closet_dist
        """
        # TODO: instead of camera focal point; calibrate closet pred to
        # operator view
        H, W, C = rgb_img.shape
        focal_y = H / 2
        focal_x = W / 2
        closet_pred = None
        closet_dist = None  # Cloest to camera in z-axis
        closet_dist = 99999999
        for bbox in bbox_list:
            center_x, center_y, width, height = bbox
            cur_dist = (center_x - focal_x)**2 + (center_y - focal_y)**2
            if closet_pred is None:
                closet_pred = bbox
                closet_dist = cur_dist
            else:
                if cur_dist < closet_dist:
                    closet_pred = bbox
                    closet_dist = cur_dist
        return closet_pred, closet_dist

    def get_rotation_angle(self, bbox_center_x, bbox_center_y):
        """Given a bounding box center, return the yaw/pitch difference in radians.

        Args:
            bbox_center_x (float): x coordinate of the center of the bounding box
            bbox_center_y (float): y coordinate of the center of the bounding box

        Returns:
            (float, float): yaw_diff, pitch_diff in radians
        """
        yaw_diff = (bbox_center_x - self.CFG.IMG_CENTER_X) * \
            (self.CFG.AUTOAIM_CAMERA.YAW_FOV_HALF / self.CFG.IMG_CENTER_X)
        pitch_diff = (bbox_center_y - self.CFG.IMG_CENTER_Y) * \
            (self.CFG.AUTOAIM_CAMERA.PITCH_FOV_HALF / self.CFG.IMG_CENTER_Y)

        return yaw_diff, pitch_diff
