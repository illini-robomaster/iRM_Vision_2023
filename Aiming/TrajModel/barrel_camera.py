"""Host the method that compute camera to barrel transform."""

import numpy as np


def get_camera_barrel_T(cfg):
    """Compute camera to barrel transform.

    Args:
        cfg (python object): config.py config node object

    Returns:
        np.array: SE(3) 4x4 transform matrix
    """
    # TODO(roger): support only steering robot now
    x_offset = 0.1
    y_offset = 0.05
    z_offset = 0.085
    camera_barrel_T = np.eye(4)

    return camera_barrel_T
