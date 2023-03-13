"""Defines a basic linear tracker and base classes for future dev."""
import scipy.optimize
import numpy as np

from .consistent_id_gen import ConsistentIdGenerator

# TODO: move this to config
FRAME_BUFFER_SIZE = 10

# TODO: this class should be part of abstract base tracker class
class tracked_armor(object):
    """A class that represents a tracked armor.

    It stores the history of bounding boxes and ROIs, and can predict the
    bounding box of the next frame.
    """

    def __init__(self, bbox, roi, frame_tick, armor_id):
        """Initialize from prediction.
        
        Args:
            bbox (tuple): (center_x, center_y, w, h)
            roi (np.ndarray): ROI of the armor
            frame_tick (int): frame tick
            armor_id (int): unique ID
        """
        self.bbox_buffer = [bbox]
        self.roi_buffer = [roi]
        self.observed_frame_tick = [frame_tick]
        self.armor_id = armor_id  # unique ID

    def compute_cost(self, other_armor):
        """Compute the cost of matching this armor with another armor.
        
        Args:
            other_armor (tracked_armor): another armor
        
        Returns:
            float: cost
        """
        assert isinstance(other_armor, tracked_armor)
        # TODO: use more sophisticated metrics (e.g., RGB) as cost function
        c_x, c_y, w, h = self.bbox_buffer[-1]
        o_c_x, o_c_y, o_w, o_h = other_armor.bbox_buffer[-1]
        return np.square(c_x - o_c_x) + np.square(c_y - o_c_y)

    def update(self, other_armor, frame_tick):
        """Update the state of this armor with matched armor.

        Args:
            other_armor (tracked_armor): another armor
            frame_tick (int): frame tick
        """
        # Only call if these two armors are matched
        self.bbox_buffer.append(other_armor.bbox_buffer[-1])
        self.roi_buffer.append(other_armor.roi_buffer[-1])
        self.observed_frame_tick.append(frame_tick)

        # Maintain each armor's buffer so that anything older than
        # FRAME_BUFFER_SIZE is dropped
        self.bbox_buffer = self.bbox_buffer[-FRAME_BUFFER_SIZE:]
        self.roi_buffer = self.roi_buffer[-FRAME_BUFFER_SIZE:]

    def predict_bbox(self, cur_frame_tick):
        """Predict the bounding box of the tracked armor at cur frame tick.
        
        Args:
            cur_frame_tick (int): current frame tick
        
        TODO
            - Use Kalman filter to do prediction
            - Support future frame idx for predictions

        Returns:
            tuple: (center_x, center_y, w, h)
        """
        if cur_frame_tick == self.observed_frame_tick[-1] or len(
                self.bbox_buffer) == 1:
            return self.bbox_buffer[-1]
        else:
            # Linear extrapolation
            c_x, c_y, w, h = self.bbox_buffer[-1]
            o_c_x, o_c_y, o_w, o_h = self.bbox_buffer[-2]
            delta_tick = self.observed_frame_tick[-1] - \
                self.observed_frame_tick[-2]
            new_delta_tick = cur_frame_tick - self.observed_frame_tick[-1]
            delta_x = (c_x - o_c_x) * new_delta_tick / delta_tick
            delta_y = (c_y - o_c_y) * new_delta_tick / delta_tick
            return (int(c_x + delta_x), int(c_y + delta_y), w, h)


class basic_tracker(object):
    """
    Basic tracker that can handle only one target.

    It memorizes the state of last two predictions and do linear extrapolation
    """

    SE_THRESHOLD = 3200  # (40, 40) pixels away

    def __init__(self, config):
        """Initialize the simple lineartracker.
        
        Args:
            config (python object): shared config
        """
        self.CFG = config
        self.active_armors = []
        self.id_gen = ConsistentIdGenerator()
        self.frame_tick = 0  # TODO: use timestamp may be a better idea

    def process_one(self, pred_list, enemy_team, rgb_img):
        """Process one set of detections and rgb images.

        Args:
            pred_list (list): list of (name, conf, bbox) tuples
            enemy_team (str): enemy team name (blue or red)
            rgb_img (np.ndarray): RGB image

        Returns:
            (list, list): list of tracked_armors (detections+predictions) and their IDs
        """
        new_armors = []
        for name, conf, bbox in pred_list:
            c_x, c_y, w, h = bbox
            lower_x = int(c_x - w / 2)
            upper_x = int(c_x + w / 2)
            lower_y = int(c_y - h / 2)
            upper_y = int(c_y + h / 2)
            roi = rgb_img[lower_y:upper_y, lower_x:upper_x]
            new_armors.append(tracked_armor(bbox, roi, self.frame_tick, -1))

        if len(self.active_armors) > 0:
            # Try to associate with current armors
            cost_matrix = np.zeros(
                (len(new_armors), len(self.active_armors)), dtype=float)
            for i in range(len(new_armors)):
                for j in range(len(self.active_armors)):
                    cost_ij = new_armors[i].compute_cost(self.active_armors[j])
                    cost_matrix[i, j] = cost_ij
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                cost_matrix)

            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.SE_THRESHOLD:
                    assert new_armors[i] is not None
                    self.active_armors[j].update(
                        new_armors[i], self.frame_tick)
                    new_armors[i] = None

        new_armors = [i for i in new_armors if i is not None]

        for n in new_armors:
            n.armor_id = self.id_gen.get_id()

        # Maintain current buffer. If an armor has not been observed by 10
        # frames, it is dropped
        self.active_armors = [i for i in self.active_armors
                              if self.frame_tick - i.observed_frame_tick[-1] < FRAME_BUFFER_SIZE]

        # Unassociated armors are registered as new armors
        for a in new_armors:
            self.active_armors.append(a)

        # Create a list of bbox and unique IDs to return
        ret_bbox_list = []
        ret_id_list = []
        for a in self.active_armors:
            # If an armor is observed, directly use the bbox
            ret_bbox_list.append(a.predict_bbox(self.frame_tick))
            ret_id_list.append(a.armor_id)

        self.frame_tick += 1

        return ret_bbox_list, ret_id_list
