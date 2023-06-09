"""Implemented EKF tracker with filterpy library."""
import scipy.optimize
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
from .consistent_id_gen import ConsistentIdGenerator

# TODO: move this to config
FRAME_BUFFER_SIZE = 10


class KalmanTracker(object):
    """A class that represents a Kalman Fitler tracker.

    It is the matrix for KF tracking that will be stored in each armor.
    """

    def __init__(self):
        """Initialize EKF from armor."""
        dt = 1
        self.kalman = ExtendedKalmanFilter(dim_x=6, dim_z=2)
        # F:transition matrix
        self.kalman.F = np.array([[1, 0, dt, 0, 0.5 * (dt**2), 0],
                                  [0, 1, 0, dt, 0, 0.5 * (dt**2)],
                                  [0, 0, 1, 0, dt, 0],
                                  [0, 0, 0, 1, 0, dt],
                                  [0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 1]], np.float32)
        # x:initial state初始状态 initialize the KF position to center
        self.kalman.x = np.array([320, 180, 0, 0, 0, 0])
        # R:measurement noise matrix
        # self.kalman.R *= 10
        self.kalman.R = np.array([[1, 0], [0, 1]], np.float32) * 1
        # Q:process noise matrix
        self.kalman.Q = np.eye(6)
        # P:初始协方差矩阵, 初始状态很不确定时把这个拧大一点
        self.kalman.P *= 10
        # measurement and prediction
        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)

    def H_of(self, x):
        """Compute the jacobian of the measurement function."""
        # Jocobian of the measurement function H
        # The measurement model is linear, so the H matrix is
        # actually constant and independent of x
        return np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0]])

    def hx(self, x):
        """Compute the measurement from the state vector x."""
        # Return x and y directly as the measurement
        return np.array([x[0], x[1]])

    def update(self, x, y):
        """Update the state of this armor with matched armor."""
        self.measurement = np.array([x, y], np.float32)
        self.kalman.update(self.measurement, self.H_of, self.hx)
        self.kalman.predict()
        self.prediction = self.kalman.x

    def get_prediction(self):
        """Predict the x and y values.

        Returns:
            tuple: predicted x and y values
        """
        return int(self.prediction[0]), int(self.prediction[1])


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
        self.KF_matrix = KalmanTracker()

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
            # print(self.bbox_buffer[-1])
            c_x, c_y, w, h = self.bbox_buffer[-1]
            self.KF_matrix.update(c_x, c_y)
            return self.bbox_buffer[-1]
        else:
            # KF tracking
            c_x, c_y, w, h = self.bbox_buffer[-1]
            predicted_x, predicted_y = self.KF_matrix.get_prediction()
            self.KF_matrix.update(predicted_x, predicted_y)
            return (int(predicted_x), int(predicted_y), w, h)
