"""Transform a point in barrel's coordinate system to robot coordinate system."""

import numpy as np
import Utils


def barrel_to_robot_T(gimbal_yaw, gimbal_pitch, armor_xyz):
    """Transform a point in barrel's coordinate system to robot coordinate system.

    The key difference is to offset gimbal's rotation.

    Args:
        gimbal_yaw (float): yaw angle of gimbal
        gimbal_pitch (float): pitch angle of gimbal
        armor_xyz (np.array): point in barrel's coordinate system

    Returns:
        np.array: point in robot coordinate system
    """
    r, theta, phi = Utils.cartesian_to_spherical(*list(armor_xyz.flatten()))
    phi += gimbal_yaw
    theta += gimbal_pitch
    x, y, z = Utils.spherical_to_cartesian(r, theta, phi)
    return x, y, z
