"""Per-robot EKF tracker based on RM_Vision."""
import numpy as np

from .consistent_id_gen import ConsistentIdGenerator
from .EKF_filter import ExtendedKalmanFilter

# TODO: move this to config
FRAME_BUFFER_SIZE = 1

class tracker(object):
    """
    Basic tracker that can handle only one target.

    It memorizes the state of last two predictions and do linear extrapolation
    """

    SE_THRESHOLD = 0.1

    def __init__(self, config):
        """Initialize the simple lineartracker.

        Args:
            config (python object): shared config
        """
        self.CFG = config
        self.tracked_id = None
        # Possible states: LOST, DETECTING, TRACKING, TEMP_LOST
        self.tracked_state = 'LOST'
        self.measurement = np.zeros(4)
        self.target_state = np.zeros(9)
        self.max_match_distance_ = 0.5

        # Different from rm_vision with wrapping fix
        self.max_match_yaw_diff_ = np.pi / 4  # 45 degrees
        self.last_yaw_ = 0
        self.detect_count_ = 0
        self.lost_count_ = 0
        self.ekf = ExtendedKalmanFilter()

        # 'NORMAL', 'BALANCED', 'OUTPOST'
        self.tracked_armors_num = 'NORMAL'
        # self.id_gen = ConsistentIdGenerator()
        # self.frame_tick = 0  # TODO: use timestamp may be a better idea
    
    def update_armors_num(self):
        # FIXME: implement this to adapt to different armor boards for RMUC!
        raise NotImplementedError
    
    def init_tracker_(self, pred_list):
        closet_distance = 99999999
        selected_armor_idx = -1
        if len(pred_list) == 0:
            return
        for armor_idx in range(len(pred_list)):
            armor_type, armor_xyz, bbox, armor_yaw = pred_list[armor_idx]
            c_x, c_y, _, _ = bbox
            distance = np.sqrt(c_x ** 2 + c_y ** 2)
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
        ekf_pred = self.ekf.predict(dt)

        matched = False
        self.target_state = ekf_pred

        if len(pred_list) > 0:
            same_id_armor = None
            same_id_armors_count = 0
            yaw_diff = 99999999
            min_position_diff = 99999999
            predicted_armor_xyz = self.get_armor_position_from_state(ekf_pred)
            for armor_idx in range(len(pred_list)):
                armor_type, armor_xyz, bbox, armor_yaw = pred_list[armor_idx]
                if armor_type != self.tracked_id:
                    continue
                same_id_armor = pred_list[armor_idx]
                same_id_armors_count += 1
                position_diff = np.linalg.norm(predicted_armor_xyz - armor_xyz)
                if position_diff < min_position_diff:
                    min_position_diff = position_diff
                    self.tracked_armor = pred_list[armor_idx]
                    yaw_diff = np.abs(self.orientation_to_yaw(armor_yaw) - ekf_pred[6]) % (2 * np.pi)
            
            if min_position_diff < self.max_match_distance_ and yaw_diff < self.max_match_yaw_diff_:
                matched = True
                # Update EKF
                measured_yaw = self.orientation_to_yaw(self.tracked_armor[3])
                mesuared_xyz = self.tracked_armor[1]
                self.measurement = np.array([mesuared_xyz[0], mesuared_xyz[1], mesuared_xyz[2], measured_yaw])
                self.target_state = self.ekf.update(self.measurement, dt)
            elif same_id_armors_count == 1 and yaw_diff > self.max_match_yaw_diff_:
                self.handle_armor_jump_(same_id_armor)
        
        # Ad-hoc post processing
        if self.target_state[8] < 0.12:
            self.target_state[8] = 0.12
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
        # reset EKF state for new armor
        armor_type, armor_xyz, bbox, armor_yaw = same_id_armor
        yaw = self.orientation_to_yaw(armor_yaw)
        self.target_state[6] = yaw

        # FIXME
        # self.update_armors_num()
        if self.tracked_armors_num == 'NORMAL':
            dz = self.target_state[4] - armor_xyz[2]
            self.target_state[4] = armor_xyz[2]
            self.another_r, self.target_state[8] = self.target_state[8], self.another_r
        
        infer_p = self.get_armor_position_from_state(self.target_state)
        if np.linalg.norm(infer_p - armor_xyz) > self.max_match_distance_:
            r = self.target_state[8]
            self.target_state[0] = armor_xyz[0] + r * np.cos(yaw)
            self.target_state[1] = 0
            self.target_state[2] = armor_xyz[1] + r * np.sin(yaw)
            self.target_state[3] = 0
            self.target_state[4] = armor_xyz[2]
            self.target_state[5] = 0
        
        self.ekf.set_state(self.target_state)
    
    def orientation_to_yaw(self, yaw):
        # shortest distance to last_yaw_
        yaw_diff = yaw - self.last_yaw_
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        elif yaw_diff < -np.pi:
            yaw_diff += 2 * np.pi
        # TODO(roger): it is a bad practice to modify class variable in a utility function
        new_yaw = self.last_yaw_ + yaw_diff
        self.last_yaw_ = yaw
        return new_yaw

    def get_armor_position_from_state(self, x):
        xc = x[0]
        yc = x[2]
        za = x[4]
        yaw = x[6]
        r = x[8]
        xa = xc - r * np.cos(yaw)
        ya = yc - r * np.sin(yaw)
        return np.array([xa, ya, za])
    
    def init_EKF(self, pred_list, selected_armor_idx):
        armor_type, armor_xyz, bbox, armor_yaw = pred_list[selected_armor_idx]
        xa = armor_xyz[0]
        ya = armor_xyz[1]
        za = armor_xyz[2]
        self.last_yaw_ = 0
        yaw = self.orientation_to_yaw(armor_yaw)

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

                gimbal_pitch = np.arctan2(armor_pos[2], armor_pos[0])
                gimbal_yaw = -np.arctan2(armor_pos[1], armor_pos[0])

                return dist, gimbal_pitch, gimbal_yaw
