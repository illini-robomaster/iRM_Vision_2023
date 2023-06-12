"""Implemented EKF tracker with filterpy library."""
import scipy.optimize
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
from .consistent_id_gen import ConsistentIdGenerator

import Utils

# TODO: move this to config
FRAME_BUFFER_SIZE = 10


class KalmanTracker(object):
    """A class that represents a Kalman Fitler tracker.

    It is the matrix for KF tracking that will be stored in each armor.
    """

    def __init__(self, init_pitch, init_yaw):
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
        # x:initial state initialize the KF position to center
        self.kalman.x = np.array([init_pitch, init_yaw, 0, 0, 0, 0])
        # R:measurement noise matrix
        # self.kalman.R *= 10
        self.kalman.R = np.array([[1, 0], [0, 1]], np.float32) * 1
        # Q:process noise matrix
        self.kalman.Q = np.eye(6)
        # P:initial covariance matrix. Tune this up if initial state is not accurate
        self.kalman.P *= 10
        # measurement and prediction
        self.measurement = np.array([init_pitch, init_yaw], np.float32)
        self.prediction = np.array([init_pitch, init_yaw], np.float32)

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

    def update(self, x, y, certain_flag=True):
        """Update the state of this armor with matched armor.

        Args:
            x: x state (2d coordinates)
            y: y state (2d coordinates)
            certain_x (bool): whether the x value is certain (observed)
        """
        self.measurement = np.array([x, y], np.float32)
        self.kalman.update(self.measurement, self.H_of, self.hx)
        self.kalman.predict()
        self.prediction = self.kalman.x

    def get_prediction(self):
        """Predict the x and y values.

        Returns:
            tuple: predicted x and y values
        """
        return self.prediction[0], self.prediction[1]


class tracked_armor(object):
    """A class that represents a tracked armor.

    It stores the history of bounding boxes and ROIs, and can predict the
    bounding box of the next frame.
    """

    def __init__(self, armor_type, abs_yaw, abs_pitch, y_distance, z_distance, frame_tick):
        """Initialize from prediction.

        Args:
            bbox (tuple): (center_x, center_y, w, h)
            roi (np.ndarray): ROI of the armor
            frame_tick (int): frame tick
            armor_id (int): unique ID
        """
        self.pitch_buffer = [abs_pitch]
        self.yaw_buffer = [abs_yaw]
        self.armor_type = armor_type
        self.y_distance_buffer = [y_distance]
        self.z_distance_buffer = [z_distance]
        self.observed_frame_tick = [frame_tick]
        self.armor_id = -1  # unique ID
        self.KF_matrix = KalmanTracker(abs_pitch, abs_yaw)

    def compute_cost(self, other_armor):
        """Compute the cost of matching this armor with another armor.

        Args:
            other_armor (tracked_armor): another armor

        Returns:
            float: cost
        """
        assert isinstance(other_armor, tracked_armor)
        if self.armor_type != other_armor.armor_type:
            return 99999999
        # TODO: use more sophisticated metrics (e.g., RGB) as cost function
        my_yaw = self.yaw_buffer[-1]
        my_pitch = self.pitch_buffer[-1]
        other_yaw = other_armor.yaw_buffer[-1]
        other_pitch = other_armor.pitch_buffer[-1]
        return Utils.get_radian_diff(my_yaw, other_yaw) + \
            Utils.get_radian_diff(my_pitch, other_pitch)

    def update(self, other_armor, frame_tick):
        """Update the state of this armor with matched armor.

        Args:
            other_armor (tracked_armor): another armor
            frame_tick (int): frame tick
        """
        # Only call if these two armors are matched
        assert len(other_armor.pitch_buffer) == 1
        self.pitch_buffer.append(other_armor.pitch_buffer[-1])
        self.yaw_buffer.append(other_armor.yaw_buffer[-1])
        self.y_distance_buffer.append(other_armor.y_distance_buffer[-1])
        self.z_distance_buffer.append(other_armor.z_distance_buffer[-1])
        self.observed_frame_tick.append(frame_tick)

        # Maintain each armor's buffer so that anything older than
        # FRAME_BUFFER_SIZE is dropped
        self.pitch_buffer = self.pitch_buffer[-FRAME_BUFFER_SIZE:]
        self.yaw_buffer = self.yaw_buffer[-FRAME_BUFFER_SIZE:]
        self.y_distance_buffer = self.y_distance_buffer[-FRAME_BUFFER_SIZE:]
        self.z_distance_buffer = self.z_distance_buffer[-FRAME_BUFFER_SIZE:]
        self.observed_frame_tick = self.observed_frame_tick[-FRAME_BUFFER_SIZE:]

    def predict_distance_angle(self, cur_frame_tick):
        """Predict the distance and angle of the tracked armor at cur frame tick.

        Args:
            cur_frame_tick (int): current frame tick

        Returns:
            tuple: (y_distance, z_distance, yaw, pitch)
        """
        if cur_frame_tick == self.observed_frame_tick[-1] or len(self.y_distance_buffer) == 1:
            cur_pitch = self.pitch_buffer[-1]
            cur_yaw = self.yaw_buffer[-1]
            self.KF_matrix.update(cur_pitch, cur_yaw)
            return self.y_distance_buffer[-1], self.z_distance_buffer[-1], cur_pitch, cur_yaw
        else:
            # KF tracking
            predicted_pitch, predicted_yaw = self.KF_matrix.get_prediction()
            self.KF_matrix.update(predicted_pitch, predicted_yaw)
            # TODO: use filtering to predict distances as well
            latest_y_dist = self.y_distance_buffer[-1]
            latest_z_dist = self.z_distance_buffer[-1]
            return latest_y_dist, latest_z_dist, predicted_pitch, predicted_yaw
