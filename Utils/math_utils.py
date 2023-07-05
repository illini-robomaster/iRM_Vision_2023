"""Mathematical utilities."""
import numpy as np


def cartesian_to_spherical(x, y, z):
    """Convert cartesian coordinates to spherical coordinates.

    Args:
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate

    Returns:
        r (float): radius
        theta (float): polar angle
        phi (float): azimuthal angle
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(z, np.sqrt(x**2 + y**2))
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to cartesian coordinates.

    Args:
        r (float): radius
        theta (float): polar angle
        phi (float): azimuthal angle

    Returns:
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def deg_to_rad(deg):
    """Convert degree to radian."""
    return deg * ((2 * np.pi) / 360)


def rad_to_deg(rad):
    """Convert radian to degree."""
    return rad * (360. / (2 * np.pi))


def get_radian_diff(angle1, angle2):
    """Compute the abs difference between two angles in radians.

    Parameters:
    angle1 (float): First angle in radians.
    angle2 (float): Second angle in radians.

    Returns:
    float: The difference between the angles in radians, ranging from 0 to pi.
    """
    # Normalize angles to be between 0 and 2*pi
    angle1 = angle1 % (2 * np.pi)
    angle2 = angle2 % (2 * np.pi)

    # Compute the difference along the lesser arc
    diff = abs(angle1 - angle2)
    if diff > np.pi:
        diff = 2 * np.pi - diff

    return diff
