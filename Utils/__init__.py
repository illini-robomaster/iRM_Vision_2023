# Commonly used utils functions
import numpy as np

def deg_to_rad(deg):
    return deg * ((2 * np.pi) / 360)

def rad_to_deg(rad):
    return rad * (360. / (2 * np.pi))
