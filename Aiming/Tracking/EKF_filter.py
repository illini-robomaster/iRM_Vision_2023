import numpy as np


class ExtendedKalmanFilter:
    def __init__(self):
        P0 = np.eye(9)
        # P_post: 后验误差协方差矩阵，表示状态估计的不确定性
        self.P_post = P0
        # n: 系统的维度
        self.n = P0.shape[0]
        # I: 单位矩阵
        self.I = np.eye(self.n)
        # x_post: 后验状态，表示经过更新步骤后的状态估计值
        self.x_post = np.zeros(self.n)
    
    def update_R(self, z):
        r_xyz_factor = 4e-4
        r_yaw = 5e-3
        r = np.zeros((4, 4))
        x = r_xyz_factor
        r[0, 0] = np.abs(x * z[0])
        r[1, 1] = np.abs(x * z[1])
        r[2, 2] = np.abs(x * z[2])
        r[3, 3] = r_yaw
        return r
    
    def update_Q(self, dt):
        # Update process noise covariance matrix
        # Parameters can be found here:
        # https://gitlab.com/rm_vision/rm_vision/-/blob/main/rm_vision_bringup/config/node_params.yaml
        s2qxyz = 0.05
        x = s2qxyz
        s2qyaw = 5.0
        y = s2qyaw
        s2qr = 80.0
        r = s2qr
        q_x_x = np.power(dt, 4) / 4 * x
        q_x_vx = np.power(dt, 3) / 2 * x
        q_vx_vx = np.power(dt, 2) * x
        q_y_y = np.power(dt, 4) / 4 * y
        # TODO(roger): this is different from the c++ implementation
        # https://gitlab.com/rm_vision/rm_auto_aim/-/blob/main/armor_tracker/src/tracker_node.cpp#L85
        q_y_vy = np.power(dt, 3) / 2 * y
        q_vy_vy = np.power(dt, 2) * y
        q_r = np.power(dt, 4) / 4 * r
        q_mat = np.array([
            [q_x_x, q_x_vx, 0, 0, 0, 0, 0, 0, 0],
            [q_x_vx, q_vx_vx, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, q_x_x, q_x_vx, 0, 0, 0, 0, 0],
            [0, 0, q_x_vx, q_vx_vx, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, q_x_x, q_x_vx, 0, 0, 0],
            [0, 0, 0, 0, q_x_vx, q_vx_vx, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, q_y_y, q_y_vy, 0],
            [0, 0, 0, 0, 0, 0, q_y_vy, q_vy_vy, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, q_r]
        ])
        return q_mat
    
    def get_jacobian_f(self, x, dt):
        # Jacobian of state transition function
        jacobian_f = np.eye(self.n)
        jacobian_f[0, 1] = dt
        jacobian_f[2, 3] = dt
        jacobian_f[4, 5] = dt
        jacobian_f[6, 7] = dt
        return jacobian_f

    def get_jacobian_h(self, x, dt):
        # Jacobian of observation function
        jacobian_h = np.zeros((4, 9))
        yaw = x[6]
        r = x[8]
        jacobian_h[0, 0] = 1
        jacobian_h[0, 6] = r * np.sin(yaw)
        jacobian_h[0, 8] = -np.cos(yaw)
        jacobian_h[1, 2] = 1
        jacobian_h[1, 6] = -r * np.cos(yaw)
        jacobian_h[1, 8] = -np.sin(yaw)
        jacobian_h[2, 4] = 1
        jacobian_h[3, 6] = 1
        return jacobian_h
    
    def f(self, state, dt):
        # EKF
        # xa = x_armor, xc = x_robot_center
        # state: xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r
        # measurement: xa, ya, za, yaw
        # dt: current observation time - previous observation time
        # f - Process function
        state_new = state.copy()
        assert len(state) == 9
        state_new[0] += state[1] * dt
        state_new[2] += state[3] * dt
        state_new[4] += state[5] * dt
        state_new[6] += state[7] * dt
        return state_new

    def h(self, state):
        # observation function
        z = np.zeros(4)
        xc = state[0]
        yc = state[2]
        yaw = state[6]
        r = state[8]
        z[0] = xc - r * np.cos(yaw)
        z[1] = yc - r * np.sin(yaw)
        z[2] = state[4]
        z[3] = yaw
        return z

    def set_state(self, x0):
        # Overwrite the initial state
        self.x_post = x0

    def predict(self, dt):
        # Predict next state
        self.F = self.get_jacobian_f(self.x_post, dt)
        self.Q = self.update_Q(dt)

        self.x_pri = self.f(self.x_post)  # 通过状态转移函数进行预测
        self.P_pri = self.F @ self.P_post @ self.F.T + self.Q  # 更新先验误差协方差矩阵

        # 对下一次预测准备
        self.x_post = self.x_pri
        self.P_post = self.P_pri

        return self.x_pri

    def update(self, z, dt):
        # 更新步骤
        self.H = self.get_jacobian_h(self.x_pri, dt)
        self.R = self.update_R(z)

        # 计算卡尔曼增益
        self.K = self.P_pri @ self.H.T @ np.linalg.inv(self.H @ self.P_pri @ self.H.T + self.R)
        self.x_post = self.x_pri + self.K @ (z - self.h(self.x_pri))  # 用观测数据更新状态预测
        self.P_post = (self.I - self.K @ self.H) @ self.P_pri  # 更新后验误差协方差矩阵

        return self.x_post
