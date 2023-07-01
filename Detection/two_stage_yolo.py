"""Two stage YOLO detector."""
import os
import onnx
import onnxruntime
import numpy as np
import cv2

from Detection.yolo import yolo_detector


class armor_class:
    """Placeholder armor struct to replace traditional armors."""

    def __init__(self, left_light, right_light, bbox, conf, cls):
        """Initialize the armor struct.

        Args:
            left_light (light_class): left light bar
            right_light (light_class): right light bar
            bbox (np.array): bounding box of the armor; in order of [min_x, min_y, max_x, max_y]
            conf (float): confidence of the armor
        """
        self.left_light = left_light
        self.right_light = right_light
        self.conf = conf
        self.cls = cls
        # Transform bbox to center, width, height
        self.bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        self.bbox_width = bbox[2] - bbox[0]
        self.bbox_height = bbox[3] - bbox[1]
        self.bbox = np.array([
            self.bbox_center[0],
            self.bbox_center[1],
            self.bbox_width,
            self.bbox_height
        ])


class two_stage_yolo_detector:
    """Two stage YOLO detector.

    First stage: YOLOv7 with shufflenet backbone
        - Timing
            - 0.03s (CPU ONNX) on M1 Pro MacBook Pro
            - 0.002s (FP32 TRT) on RTX 3090
            - 0.01s (FP32 TRT) on Orin NANO
    Second stage: light bar detection for PnP
        - Timing
            - 0.0002s on M1 Pro MacBook Pro
    """

    def __init__(self, cfg, init_enemy_team):
        """Initialize the two stage YOLO detector.

        Args:
            cfg (node): python config node object
            init_enemy_team (str): initial enemy team color
        """
        self.CFG = cfg
        self.change_target_color(init_enemy_team)
        self.yolo = yolo_detector(cfg)

    def change_target_color(self, new_color):
        """Change the target color of the detector.

        Args:
            new_color (str): enemy team color ['red', 'blue']
        """
        assert new_color in ['red', 'blue']
        self.target_color = new_color
        if self.target_color == 'red':
            self.target_color_prefix = 'R'
        elif self.target_color == 'blue':
            self.target_color_prefix = 'B'
        else:
            raise ValueError('Invalid color: ' + new_color)

    def detect(self, resized_bgr_frame, raw_bgr_frame=None):
        """Detect armors in the frame.

        Args:
            resized_bgr_frame (np.array): BGR frame with reduced size
            raw_bgr_frame (np.array, optional): raw BGR frame full resolution. Defaults to None.

        Returns:
            armor_list: list of armor_class objects
        """
        if raw_bgr_frame is None:
            raw_bgr_frame = resized_bgr_frame

        resized_rgb_frame = cv2.cvtColor(resized_bgr_frame, cv2.COLOR_BGR2RGB)

        if self.CFG.DEBUG_DISPLAY:
            viz_img = resized_bgr_frame.copy()

        pred_list = self.yolo.detect(resized_rgb_frame)

        if self.CFG.DEBUG_DISPLAY:
            for min_x, min_y, max_x, max_y, conf, cls_name in pred_list:
                viz_img = cv2.rectangle(viz_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                viz_img = cv2.putText(viz_img,
                                      cls_name,
                                      (min_x, min_y),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      1,
                                      (0, 255, 0),
                                      2,
                                      cv2.LINE_AA)
        
        if self.CFG.DEBUG_DISPLAY:
            # write current enemy team color on the frame
            viz_img = cv2.putText(viz_img,
                                    self.target_color,
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA)
            cv2.imshow('yolo', viz_img)
            cv2.waitKey(1)

        pred_list = [pred for pred in pred_list if pred[5].startswith(self.target_color_prefix)]

        armor_list = []

        for min_x, min_y, max_x, max_y, conf, cls in pred_list:
            # TEST COLOR; INCLUDE ONLY ENEMY
            bbox = np.array([min_x, min_y, max_x, max_y])
            roi_img = resized_rgb_frame[min_y:max_y, min_x:max_x]
            gray_img = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
            # TODO: use raw image to further increase PnP precision
            thres, binary_img = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY)
            bin_contours, _ = cv2.findContours(
                binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bin_contours = [ctr for ctr in bin_contours if ctr.shape[0] >= 5]

            if len(bin_contours) < 2:
                continue

            light_list = []

            for contour in bin_contours:
                light = cv2.minAreaRect(contour)
                light = light_class(light)

                try:
                    if not light.is_light():
                        continue
                except BaseException:
                    import pdb
                    pdb.set_trace()

                light.offset_bbox(min_x, min_y)

                # TODO: add color tests?
                light_list.append(light)

            if len(light_list) < 2:
                continue

            if light_list[0].center[0] < light_list[1].center[0]:
                left_light = light_list[0]
                right_light = light_list[1]
            else:
                left_light = light_list[1]
                right_light = light_list[0]

            armor = armor_class(left_light, right_light, bbox, conf, cls)
            armor_list.append(armor)

        return armor_list


class light_class:
    """
    A class that represents a light bar on the armor board.

    Most importantly, given a rotated rect from OpenCV, it can calculate the
    center, top, bottom, length, width, and tilt angle of the light bar.

    For instance, OpenCV could say a rotated rect is 180 degrees tilted, but
    in fact, it is 0 degrees tilted. This class can correct the tilt angle.

    Attributes:
        center (np.ndarray): center of the light bar
        top (np.ndarray): top point of the light bar
        btm (np.ndarray): bottom point of the light bar
        length (float): length of the light bar
        width (float): width of the light bar
        tilt_angle (float): tilt angle of the light bar
    """

    LIGHT_MIN_RATIO = 0.1
    LIGHT_MAX_RATIO = 0.55
    LIGHT_MAX_ANGLE = 40.0

    def __init__(self, rotated_rect):
        """Initialize the light bar.

        Args:
            rotated_rect (tuple): (center, (width, height), angle)
        """
        ((self.center_x, self.center_y), (_, _), _) = rotated_rect
        # sort according to y
        pts = sorted(cv2.boxPoints(rotated_rect), key=lambda x: x[1])
        self.top = (pts[0] + pts[1]) / 2
        self.btm = (pts[2] + pts[3]) / 2

        self.length = cv2.norm(self.top - self.btm)
        self.width = cv2.norm(pts[0] - pts[1])

        self.tilt_angle = np.arctan2(
            np.abs(
                self.top[0] -
                self.btm[0]),
            np.abs(
                self.top[1] -
                self.btm[1]))
        self.tilt_angle = self.tilt_angle / np.pi * 180

        self.center = np.array([self.center_x, self.center_y])

    def offset_bbox(self, min_x, min_y):
        """Offset the light bar by amount of its origin bbox.

        Args:
            min_x (float): leftmost x of the bbox
            min_y (float): top y of the bbox
        """
        self.center_x += min_x
        self.center_y += min_y

        self.top += np.array([min_x, min_y])
        self.btm += np.array([min_x, min_y])
        self.center += np.array([min_x, min_y])

    def is_light(self):
        """Apply filtering to determine if a light bar is valid.

        Criteria:
            1. Aspect ratio can not be too large or too small (light bars are slim verticals)
            2. Tilt angle (light bars are vertical)

        Args:
            light (light_class): light bar to be filtered

        Returns:
            bool: True if the light bar is valid
        """
        ratio = self.width / self.length
        ratio_ok = self.LIGHT_MIN_RATIO < ratio and ratio < self.LIGHT_MAX_RATIO

        angle_ok = self.tilt_angle < self.LIGHT_MAX_ANGLE

        return ratio_ok and angle_ok
