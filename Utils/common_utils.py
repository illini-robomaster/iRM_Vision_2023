"""Common utility functions (pathing / params choosing)."""
import numpy as np


def get_intrinsic_matrix(cfg):
    """Get the camera intrinsic matrix.

    Args:
        cfg (object): config.py node object

    Returns:
        K: camera intrinsic matrix 3x3
    """
    if hasattr(cfg, 'K'):
        return cfg.K
    elif hasattr(cfg.AUTOAIM_CAMERA, 'K'):
        return cfg.AUTOAIM_CAMERA.K
    else:
        # Infer from image size
        K = np.eye(3)

        H = cfg.IMG_HEIGHT
        W = cfg.IMG_WIDTH
        fl = min(H, W) * 2
        cx = W / 2
        cy = H / 2

        K[0, 0] = fl
        K[1, 1] = fl
        K[0, 2] = cx
        K[1, 2] = cy

        return K
