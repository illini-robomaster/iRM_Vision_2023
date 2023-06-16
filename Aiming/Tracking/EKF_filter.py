import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, f, h, jacobian_f, jacobian_h, update_Q, update_R, P0):
        # f: 状态转移函数
        self.f = f
        # h: 观测函数
        self.h = h
        # jacobian_f: 状态转移函数的雅克比矩阵
        self.jacobian_f = jacobian_f
        # jacobian_h: 观测函数的雅克比矩阵
        self.jacobian_h = jacobian_h
        # update_Q: 更新过程噪声协方差矩阵的函数
        self.update_Q = update_Q
        # update_R: 更新观测噪声协方差矩阵的函数
        self.update_R = update_R
        # P_post: 后验误差协方差矩阵，表示状态估计的不确定性
        self.P_post = P0
        # n: 系统的维度
        self.n = P0.shape[0]
        # I: 单位矩阵
        self.I = np.eye(self.n)
        # x_post: 后验状态，表示经过更新步骤后的状态估计值
        self.x_post = np.zeros(self.n)

    def set_state(self, x0):
        # 设置初始状态
        self.x_post = x0

    def predict(self):
        # 预测步骤
        self.F = self.jacobian_f(self.x_post)
        self.Q = self.update_Q()

        self.x_pri = self.f(self.x_post)  # 通过状态转移函数进行预测
        self.P_pri = self.F @ self.P_post @ self.F.T + self.Q  # 更新先验误差协方差矩阵

        # 对下一次预测准备
        self.x_post = self.x_pri
        self.P_post = self.P_pri

        return self.x_pri

    def update(self, z):
        # 更新步骤
        self.H = self.jacobian_h(self.x_pri)
        self.R = self.update_R(z)

        # 计算卡尔曼增益
        self.K = self.P_pri @ self.H.T @ np.linalg.inv(self.H @ self.P_pri @ self.H.T + self.R)
        self.x_post = self.x_pri + self.K @ (z - self.h(self.x_pri))  # 用观测数据更新状态预测
        self.P_post = (self.I - self.K @ self.H) @ self.P_pri  # 更新后验误差协方差矩阵

        return self.x_post
