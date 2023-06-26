"""Per-robot EKF tracker based on RM_Vision."""
import numpy as np

from .consistent_id_gen import ConsistentIdGenerator
from .EKF_filter import ExtendedKalmanFilter
import Utils

class tracker:
    """
    EKF tracker that selects a single robot and tracks it (can't handle multiple robots).

    Tracking logic:
        - DETECTING: the tracker is initializing
        - TRACKING: the tracker is tracking a robot
        - TEMP_LOST: the tracker lost the robot for a short time
        - LOST: the tracker lost the robot for a long time

    TODO(roger): write more about the tracker

    FIXME:
        - orientation_to_yaw does not handle the case for transformations > np.pi * 2
        - x/y/z location tracking is extremely inaccurate with PnP; select armor with angles.
    """

    MAX_MATCH_DISTANCE_ = 0.7

    # Different from RV; fix wrapping
    MAX_MATCH_YAW_DIFF_ = 0.3

    def __init__(self, config):
        """Initialize the tracker.

        Args:
            config (python object): shared config
        """
        self.CFG = config
        self.tracked_id = None
        # Possible states: LOST, DETECTING, TRACKING, TEMP_LOST
        self.tracked_state = 'LOST'
        self.measurement = np.zeros(4)
        self.target_state = np.zeros(9)

        self.last_yaw_ = 0
        self.detect_count_ = 0
        self.lost_count_ = 0
        self.ekf = ExtendedKalmanFilter()

        # 'NORMAL', 'BALANCED', 'OUTPOST'
        self.tracked_armors_num = 'NORMAL'

    def update_armors_num(self):
        """Update armor type to handle different numbers and sizes.

        Raises:
            NotImplementedError: TBD
        """
        # FIXME: implement this to adapt to different armor boards for RMUC!
        raise NotImplementedError

    def init_tracker_(self, pred_list):
        """Initialize the tracker.

        Internal function. DO NOT call this function directly.

        Args:
            pred_list (list): a list of recognized armors
        """
        closet_distance = 99999999
        selected_armor_idx = -1
        if len(pred_list) == 0:
            return
        for armor_idx in range(len(pred_list)):
            armor_type, armor_xyz, bbox, armor_yaw = pred_list[armor_idx]
            c_x, c_y, _, _ = bbox
            distance = c_x ** 2 + c_y ** 2
            if distance < closet_distance:
                closet_distance = distance
                selected_armor_idx = armor_idx

        assert selected_armor_idx != -1

        armor_type, armor_xyz, bbox, armor_yaw = pred_list[selected_armor_idx]

        self.init_EKF(pred_list, selected_armor_idx)
        self.tracked_id = armor_type
        self.tracked_state = 'DETECTING'

        # FIXME
        # self.update_armors_num()

    def update_tracker_(self, pred_list, dt):
        """Update the tracker.

        Internal function. DO NOT call this function directly.

        Args:
            pred_list (list): a list of recognized armors
            dt (float): time interval between two observations
        """
        ekf_pred = self.ekf.predict(dt)

        matched = False
        self.target_state = ekf_pred

        same_id_armor_list = [a for a in pred_list if a[0] == self.tracked_id]
        if len(same_id_armor_list) == 3:
            print("More than 2 armors detected?")
        elif len(same_id_armor_list) == 2:
            # Seeing two armors.
            # Pick the most front-facing one
            print("Two armors")
            armor1_yaw_diff = np.abs(self.orientation_to_yaw(same_id_armor_list[0][3]) - ekf_pred[6])
            armor2_yaw_diff = np.abs(self.orientation_to_yaw(same_id_armor_list[1][3]) - ekf_pred[6])
            if armor1_yaw_diff < armor2_yaw_diff:
                prv_armor_idx = 0
            else:
                prv_armor_idx = 1
            if np.abs(same_id_armor_list[0][3]) < np.abs(same_id_armor_list[1][3]):
                selected_armor_idx = 0
            else:
                selected_armor_idx = 1
            if selected_armor_idx != prv_armor_idx:
                print("2 armor jump!")
                self.handle_armor_jump_(same_id_armor_list[selected_armor_idx])
            matched = True
            # Update EKF
            measured_yaw = self.orientation_to_yaw(same_id_armor_list[selected_armor_idx][3])
            self.last_yaw_ = measured_yaw
            mesuared_xyz = same_id_armor_list[selected_armor_idx][1]
            self.measurement = np.array(
                [mesuared_xyz[0], mesuared_xyz[1], mesuared_xyz[2], measured_yaw])
            self.target_state = self.ekf.update(self.measurement, dt)
        elif len(same_id_armor_list) == 1:
            armor_type, armor_xyz, bbox, armor_yaw = same_id_armor_list[0]
            yaw_diff = np.abs(self.orientation_to_yaw(
                armor_yaw) - ekf_pred[6]) % (2 * np.pi)
            if yaw_diff > self.MAX_MATCH_YAW_DIFF_:
                self.handle_armor_jump_(same_id_armor_list[0])
            matched = True
            # Update EKF
            measured_yaw = self.orientation_to_yaw(armor_yaw)
            self.last_yaw_ = measured_yaw
            mesuared_xyz = armor_xyz
            self.measurement = np.array(
                [mesuared_xyz[0], mesuared_xyz[1], mesuared_xyz[2], measured_yaw])
            self.target_state = self.ekf.update(self.measurement, dt)

        # if len(pred_list) > 0:
        #     same_id_armor = None
        #     same_id_armors_count = 0
        #     yaw_diff = 99999999
        #     min_position_diff = 99999999
        #     predicted_armor_xyz = self.get_armor_position_from_state(ekf_pred)
        #     for armor_idx in range(len(pred_list)):
        #         armor_type, armor_xyz, bbox, armor_yaw = pred_list[armor_idx]
        #         if armor_type != self.tracked_id:
        #             continue
        #         same_id_armor = pred_list[armor_idx]
        #         same_id_armors_count += 1
        #         position_diff = np.linalg.norm(predicted_armor_xyz - armor_xyz)
        #         if position_diff < min_position_diff:
        #             min_position_diff = position_diff
        #             self.tracked_armor = pred_list[armor_idx]
        #             yaw_diff = np.abs(self.orientation_to_yaw(
        #                 armor_yaw) - ekf_pred[6]) % (2 * np.pi)

        #     if min_position_diff < self.MAX_MATCH_DISTANCE_ and yaw_diff < self.MAX_MATCH_YAW_DIFF_:
        #         matched = True
        #         # Update EKF
        #         measured_yaw = self.orientation_to_yaw(self.tracked_armor[3])
        #         mesuared_xyz = self.tracked_armor[1]
        #         self.measurement = np.array(
        #             [mesuared_xyz[0], mesuared_xyz[1], mesuared_xyz[2], measured_yaw])
        #         self.target_state = self.ekf.update(self.measurement, dt)
        #     elif same_id_armors_count == 1 and yaw_diff > self.MAX_MATCH_YAW_DIFF_:
        #         self.handle_armor_jump_(same_id_armor)

        # Ad-hoc post processing
        if self.target_state[8] < 0.2:
            self.target_state[8] = 0.2
            self.ekf.set_state(self.target_state)
        elif self.target_state[8] > 0.4:
            self.target_state[8] = 0.4
            self.ekf.set_state(self.target_state)

        # Tracking FSM
        if self.tracked_state == 'DETECTING':
            if matched:
                self.detect_count_ += 1
                # tracking_thres
                if self.detect_count_ > 3:
                    self.tracked_state = 'TRACKING'
                    self.detect_count_ = 0
            else:
                self.detect_count_ = 0
                self.tracked_state = 'LOST'
        elif self.tracked_state == 'TRACKING':
            if not matched:
                self.tracked_state = 'TEMP_LOST'
                self.lost_count_ += 1
        elif self.tracked_state == 'TEMP_LOST':
            if not matched:
                self.lost_count_ += 1
                # lost_thres
                if self.lost_count_ > 3:
                    self.tracked_state = 'LOST'
                    self.lost_count_ = 0
            else:
                self.tracked_state = 'TRACKING'
                self.lost_count_ = 0

    def handle_armor_jump_(self, same_id_armor):
        """Reset the EKF state when detected a different armor of the same robot.

        Args:
            same_id_armor (armor object): another armor object of the same robot
        """
        armor_type, armor_xyz, bbox, armor_yaw = same_id_armor
        print("Prv last yaw", self.last_yaw_)
        yaw = self.orientation_to_yaw(armor_yaw)
        self.last_yaw_ = yaw
        self.target_state[6] = yaw

        # FIXME
        # self.update_armors_num()
        if self.tracked_armors_num == 'NORMAL':
            dz = self.target_state[4] - armor_xyz[2]
            self.target_state[4] = armor_xyz[2]
            self.another_r, self.target_state[8] = self.target_state[8], self.another_r

        infer_p = self.get_armor_position_from_state(self.target_state)

        if np.linalg.norm(infer_p - armor_xyz) > self.MAX_MATCH_DISTANCE_:
            r = self.target_state[8]
            self.target_state[0] = armor_xyz[0] + r * np.cos(yaw)
            self.target_state[1] = 0
            self.target_state[2] = armor_xyz[1] + r * np.sin(yaw)
            self.target_state[3] = 0
            self.target_state[4] = armor_xyz[2]
            self.target_state[5] = 0

        self.ekf.set_state(self.target_state)

    def orientation_to_yaw(self, yaw):
        """Wrap input yaw angle to closet to last_yaw_ so filter does not jump.

        Args:
            yaw (float): orientation in radian

        Returns:
            float: yaw in radian closest to the last yaw
        """
        yaw_diff = yaw - self.last_yaw_  # shortest distance to last_yaw_
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        elif yaw_diff < -np.pi:
            yaw_diff += 2 * np.pi
        new_yaw = self.last_yaw_ + yaw_diff
        return new_yaw

    def get_armor_position_from_state(self, x):
        """Compute the armor position from the state.

        Args:
            x (np.array): state vector

        Returns:
            np.array: armor position in robot coordinate
        """
        xc = x[0]
        yc = x[2]
        za = x[4]
        yaw = x[6]
        r = x[8]
        xa = xc - r * np.cos(yaw)
        ya = yc - r * np.sin(yaw)
        return np.array([xa, ya, za])

    def init_EKF(self, pred_list, selected_armor_idx):
        """Initialize the EKF filter.

        Internal function. DO NOT call this function directly.

        Args:
            pred_list (list): a list of recognized armors
            selected_armor_idx (int): the index of the selected armor in the list
        """
        armor_type, armor_xyz, bbox, armor_yaw = pred_list[selected_armor_idx]
        xa = armor_xyz[0]
        ya = armor_xyz[1]
        za = armor_xyz[2]
        self.last_yaw_ = 0
        yaw = self.orientation_to_yaw(armor_yaw)
        self.last_yaw_ = yaw

        self.target_state = np.zeros(9)
        r = 0.26
        xc = xa + r * np.cos(yaw)
        yc = ya + r * np.sin(yaw)

        self.dz = 0
        self.another_r = r

        self.target_state[0] = xc
        self.target_state[2] = yc
        self.target_state[4] = za
        self.target_state[6] = yaw
        self.target_state[8] = r

        self.ekf.set_state(self.target_state)

    def process_one(self, pred_list, stm32_state_dict):
        """Process one frame from camera.

        Args:
            pred_list (list): a list of recognized armors
            stm32_state_dict (dict): a dictionary of ego-robot states

        Returns:
            tuple: (dist, gimbal_pitch, gimbal_yaw) for selected robot
                    None if no robot is selected or the tracking is lost.
        """
        if self.tracked_state == 'LOST':
            self.init_tracker_(pred_list)
            return None
        else:
            # FIXME
            dt = 1.0 / 30.0
            self.update_tracker_(pred_list, dt)
            if self.tracked_state == 'DETECTING':
                return None
            elif self.tracked_state == 'TRACKING' or self.tracked_state == 'TEMP_LOST':
                ret = self.target_state
                # X: from barrel rear to front
                # Y: from barrel left to right
                # Z: up is gravity direction
                armor_pos = self.get_armor_position_from_state(ret)
                dist = np.sqrt(armor_pos[0]**2 + armor_pos[2]**2)

                _, gimbal_pitch, gimbal_yaw = Utils.cartesian_to_spherical(*armor_pos)

                return dist, gimbal_pitch, gimbal_yaw
