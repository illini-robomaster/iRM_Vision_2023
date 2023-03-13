"""Host commonly used utils functions."""
import numpy as np

def deg_to_rad(deg):
    """Convert degree to radian."""
    return deg * ((2 * np.pi) / 360)

def rad_to_deg(rad):
    """Convert radian to degree."""
    return rad * (360. / (2 * np.pi))
