"""paramemters and funtions for EKF tracker"""
import numpy as np
from .EKF_filter import ExtendedKalmanFilter


class ArmorTrackerNode:
    def __init__(self):
        ...
        # Define dt here
        dt = 1

        # EKF
        # xa = x_armor, xc = x_robot_center
        # state: xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r
        # measurement: xa, ya, za, yaw

        # f - Process function
        def f(x): return x + np.array([x[1] * dt, 0, x[3] * dt, 0, x[5] * dt, 0, x[7] * dt, 0, 0])

        # J_f - Jacobian of process function
        def j_f(_): return np.array([
            [1, dt, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, dt, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, dt, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        # h - Observation function
        def h(x): return np.array(
            [x[0] - x[8] * np.cos(x[6]), x[2] - x[8] * np.sin(x[6]), x[4], x[6]])

        # J_h - Jacobian of observation function
        def j_h(x): return np.array([
            [1, 0, 0, 0, 0, 0, x[8] * np.sin(x[6]), 0, -np.cos(x[6])],
            [0, 0, 1, 0, 0, 0, -x[8] * np.cos(x[6]), 0, -np.sin(x[6])],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0]
        ])

        # update_Q - process noise covariance matrix
        # 这里c++实现如下，但我不知道python怎么用ros，就先全部设成了默认值
        # s2qxyz_ = declare_parameter("ekf.sigma2_q_xyz", 20.0);
        # s2qyaw_ = declare_parameter("ekf.sigma2_q_yaw", 100.0);
        # s2qr_ = declare_parameter("ekf.sigma2_q_r", 800.0);
        s2qxyz_ = 20.0
        s2qyaw_ = 100.0
        s2qr_ = 25.0
        Q = np.diag([s2qxyz_, s2qxyz_, s2qxyz_, s2qxyz_, s2qyaw_, s2qyaw_, s2qr_])

        # update_R - observation noise covariance matrix
        # 同update_Q,全部设成了默认值
        # s2rxyz_ = self.declare_parameter("ekf.sigma2_r_xyz", 1.0)
        # s2ryaw_ = self.declare_parameter("ekf.sigma2_r_yaw", 1.0)
        s2rxyz_ = 1.0
        s2ryaw_ = 1.0
        R = np.diag([s2rxyz_, s2rxyz_, s2rxyz_, s2ryaw_])

        self.ekf = ExtendedKalmanFilter(
            f=f,
            h=h,
            jacobian_f=j_f,
            jacobian_h=j_h,
            update_Q=Q,
            update_R=R,
            P0=9)
