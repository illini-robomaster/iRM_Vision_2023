import numpy as np
import Utils

def barrel_to_robot_T(gimbal_yaw, gimbal_pitch, armor_xyz):
    r, theta, phi = Utils.cartesian_to_spherical(*list(armor_xyz.flatten()))
    phi += gimbal_yaw
    theta += gimbal_pitch
    x, y, z = Utils.spherical_to_cartesian(r, theta, phi)
    return x, y, z
