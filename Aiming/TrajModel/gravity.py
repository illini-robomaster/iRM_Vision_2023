"""Calibrate pitch offset to account for gravity."""
import numpy as np
import numdifftools

def calibrate_pitch_gravity(cfg, x1, y1):
    """Calibrate pitch offset to account for gravity and air resistance.

    Args:
        cfg (python object): config.py config node object
        x1 (float): horizontal coordinate in hypothetical landing spot
        y1 (float): vertical coordinate in hypothetical landing spot

    Returns:
        pitch_offset (float): how much to offset pitch angle
    """
    g = cfg.GRAVITY_CONSTANT
    v = cfg.INITIAL_BULLET_SPEED
    # define the function
    # TODO(roger): solving this equation assumes no air resistance

    # You can get where this function comes from by asking GPT4 with the following prompt:

    # Consider a 2D coordinate frame where a cannon is at the origin point (0, 0).
    # Given the initial velocity of a cannon ball and its landing location, (x1, y1),
    # compute the pitch angle of the cannon.
    def f(theta):
        return x1 * np.tan(theta) - (g * x1**2) / (2 * v**2 * np.cos(theta)**2) - y1

    # define the derivative of the function
    # def df(theta):
    #     return derivative(f, theta, dx=1e-6)
    df = numdifftools.Derivative(f)

    # initial guess
    theta = 0

    # Newton's method
    success_flag = False
    for i in range(100):
        theta_new = theta - f(theta) / df(theta)
        if abs(theta_new - theta) < 1e-6:
            success_flag = True
            break
        theta = theta_new

    if success_flag:
        return theta
    else:
        return 0
