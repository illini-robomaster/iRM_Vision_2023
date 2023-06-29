import numpy as np
import time
import Utils

BUFFER_SIZE = 100
LENGTH_THRESHOLD = 5
NUM_THRESHOLD = 1

class anti_spinning :
    def __init__(self, tracked_id):
        """Get the config and initialize the anti spinning motion.

        Args:
            config (python object): shared config
        """
        self.tracked_id = tracked_id
        self.spinning_buffer = [BUFFER_SIZE]
        self.tracker_switch_length = -1
        self.tracker_switch_num = 0
        self.detect_pos = np.zeros(3)
        self.lost_pos = np.zeros(3)

    
    def is_spinning(self, pred_list):
        """
        Implement the recognization of spinning motion
        """
        buffer_length = len(self.spinning_buffer)
        if (buffer_length >= 2):
            if (self.spinning_buffer[-1] == 2 and self.spinning_buffer[-2] == 1):   
                if (abs(self.tracker_switch_length - buffer_length) <= LENGTH_THRESHOLD): 
                    self.tracker_switch_num += 1
                else:
                    self.tracker_switch_length = buffer_length
                    self.tracker_switch_num = 0
                self.spinning_buffer.clear()

            if (self.tracker_switch_num >= NUM_THRESHOLD):
                # print("Detected spinning motion!" + str(time.time()))
                # print(self.tracker_switch_num)
                return True
        return False

    def calculate_dist_pitch_yaw(self):
        """
        Calculate yaw and pitch
        """
        # print(self.detect_pos)
        # print(self.lost_pos)

        armor_pos = (self.detect_pos + self.lost_pos) / 2
        dist = np.sqrt(armor_pos[0]**2 + armor_pos[2]**2)

        _, gimbal_pitch, gimbal_yaw = Utils.cartesian_to_spherical(*armor_pos)

        return dist, gimbal_pitch, gimbal_yaw