"""Image processing utilities."""
import numpy as np
import cv2

# Consistent scratch memory
IMAGE_PADDING_TEMPLATE = np.zeros((640, 640, 3), dtype=np.uint8) + 114
IMAGE_PADDING_TEMPLATE = np.ascontiguousarray(IMAGE_PADDING_TEMPLATE)


def preprocess_img_step1(raw_img_bgr, cfg):
    """Pre-process image.

    Args:
        img (np.ndarray): BGR image
        cfg (python object): shared config object

    Returns:
        np.ndarray: pre-processed BGR image
    """
    ret_dict = {}
    if cfg.ROTATE_180:
        raw_img_bgr = np.rot90(raw_img_bgr, 2)
    assert cfg.IMG_WIDTH == raw_img_bgr.shape[1] // 2
    assert cfg.IMG_HEIGHT == raw_img_bgr.shape[0] // 2
    # This trick downsamples the image by 2x
    # It's significantly faster than general cv2.resize
    resized_img_bgr = raw_img_bgr[::2, ::2].copy()
    ret_dict['raw_img'] = raw_img_bgr
    ret_dict['resized_img_bgr'] = resized_img_bgr
    resized_img_rgb = resized_img_bgr[:, :, ::-1]
    ret_dict['resized_img_rgb'] = resized_img_rgb

    return ret_dict


def preprocess_img_step2(ret_dict, cfg):
    """Pre-process image.

    Args:
        img (np.ndarray): BGR image
        cfg (python object): shared config object

    TODO: this step is very expensive (~15ms on xavier NX). The transpose and division
    should be done in ONNX with CUDA.

    Returns:
        np.ndarray: pre-processed BGR image
    """
    resized_img_rgb = ret_dict['resized_img_rgb']
    # YOLO padding
    assert resized_img_rgb.shape[:2] == (512, 640)
    IMAGE_PADDING_TEMPLATE[64:576, :, :] = resized_img_rgb
    img = np.ascontiguousarray(
        IMAGE_PADDING_TEMPLATE.transpose(
            2, 0, 1)).astype(
        np.float32)  # to CHW

    img = img / 255.0
    img = img[None]
    ret_dict['processed_yolo_img_rgb'] = img
    return ret_dict


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
