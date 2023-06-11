"""Host commonly used utils functions."""
import numpy as np
import cv2


def deg_to_rad(deg):
    """Convert degree to radian."""
    return deg * ((2 * np.pi) / 360)


def rad_to_deg(rad):
    """Convert radian to degree."""
    return rad * (360. / (2 * np.pi))


def auto_align_brightness(img, target_v=50):
    """Standardize brightness of image.

    Args:
        img (np.ndarray): BGR image
        target_v (int, optional): target brightness. Defaults to 50.

    Returns:
        np.ndarray: BGR image with standardized brightness
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    cur_v = np.mean(v)
    v_diff = int(cur_v - target_v)

    if v_diff > 0:
        value = v_diff
        # needs lower brightness
        v[v < value] = 0
        v[v >= value] -= value
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    else:
        # brighten
        value = -v_diff
        # needs lower brightness
        v[v > (255 - value)] = 255
        v[v <= (255 - value)] += value
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img


def color_test(rgb_img, rect, color):
    """Test if the color of the roi is the same as the given color.

    Args:
        rgb_img (np.ndarray): RGB image
        rect (tuple): (x, y, w, h)
        color (int): RED or BLUE (enum)

    Returns:
        bool: True if the color of the roi is the same as the given color
    """
    # Internal ENUM
    RED = 0
    BLUE = 1

    rgb_roi = rgb_img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    sum_r = np.sum(rgb_roi[:, :, 0])
    sum_b = np.sum(rgb_roi[:, :, 2])
    if color == RED:
        return sum_r >= sum_b
    else:
        return sum_b >= sum_r


def rect_contains(rect, pt):
    """Determine if a pt is inside a rect.

    Args:
        rect (tuple): (x, y, w, h)
        pt (tuple): (x, y)

    Returns:
        bool: True if the pt is inside the rect
    """
    return rect[0] < pt[0] < rect[0] + \
        rect[2] and rect[1] < pt[1] < rect[1] + rect[3]


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
