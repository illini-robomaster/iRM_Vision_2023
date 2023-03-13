"""Host routines for computational / learning-based detections."""
import numpy as np
import cv2
import os
import time
import Utils
# Internal ENUM
RED = 0
BLUE = 1

class cv_mix_dl_detector:
    """A routine that combines CV and DL to detect armors.

    It uses CV to propose potential armors, and then uses DL to filter out false positives.
    """

    def __init__(self, config, detect_color, model_path='fc.onnx'):
        """Initialize the detector.

        Args:
            config (python object): shared config.py
            detect_color (int): RED or BLUE (enum)
            model_path (str, optional): path to the DL model. Defaults to 'fc.onnx'.
        """
        self.CFG = config
        self.armor_proposer = cv_armor_proposer(self.CFG, detect_color)
        self.digit_classifier = dl_digit_classifier(self.CFG, model_path)
        self.change_color(detect_color)

    def detect(self, rgb_img):
        """Detect armors in the given image.

        Args:
            rgb_img (np.ndarray): RGB image

        Returns:
            list: list of detected armors
        """
        return self(rgb_img)

    def __call__(self, rgb_img):
        """Detect armors in the given image.

        Args:
            rgb_img (np.ndarray): RGB image

        Returns:
            list: list of detected armors
        """
        rgb_img = Utils.auto_align_brightness(rgb_img)
        # defaults to BGR format
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # propose potential armors
        armor_list = self.armor_proposer(rgb_img)
        for armor in armor_list:
            armor.extract_number(rgb_img)

        # filter out false positives
        armor_list = self.digit_classifier(armor_list)

        # Pad into formats
        ret_list = []
        # name, conf, bbox
        # center_x, center_y, width, height = bbox
        for armor in armor_list:
            center_x, center_y = armor.center
            height = (armor.left_light.length + armor.right_light.length) / 2
            width = cv2.norm(armor.left_light.center -
                             armor.right_light.center)
            bbox = (center_x, center_y, width, height)
            name = f'armor_{self.detect_color_str}'
            conf = armor.confidence
            ret_list.append((name, conf, bbox))
        return ret_list

    def change_color(self, new_color):
        """Change the color of the detector.

        Args:
            new_color (int): RED or BLUE (enum)
        """
        if new_color in [0, 1]:
            assert new_color in [0, 1]
            self.detect_color = new_color
            self.armor_proposer.detect_color = new_color
            if new_color:
                self.detect_color_str = 'blue'
            else:
                self.detect_color_str = 'red'
        elif new_color in ['red', 'blue']:
            self.detect_color_str = new_color
            if new_color == 'red':
                color_num = 0
            else:
                color_num = 1
            self.detect_color = color_num
            self.armor_proposer.detect_color = color_num
        else:
            raise NotImplementedError

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

        self.color = None


class armor_class:
    """
    A class that represents an armor board.

    Attributes:
        left_light (light_class): left light bar
        right_light (light_class): right light bar
        center (np.ndarray): center of the armor board
        color (int): color of the armor board
        confidence (float): confidence of the armor board
        armor_type (str): 'large' or 'small'
    """

    def __init__(self, light1, light2, color):
        """Initialize the armor board.

        Args:
            light1 (light_class): left light bar
            light2 (light_class): right light bar
            color (int): color of the armor board
        """
        assert light1.color == light2.color
        assert light1.color == color
        self.color = color
        self.confidence = 0.0

        if light1.center_x < light2.center_x:
            self.left_light = light1
            self.right_light = light2
        else:
            self.left_light = light2
            self.right_light = light1

        self.center = (self.left_light.center + self.right_light.center) / 2
        self.armor_type = None  # 'large', 'small'

    def extract_number(self, rgb_img):
        """Extract number from the armor board using perspective transform.

        The perspective transform matrix is computed by using top and bottom
        points of the left and right light bars as source vertices, and
        manually set target vertices.

        Args:
            rgb_img (np.ndarray): RGB image
        """
        light_length = 12
        warp_height = 28
        small_armor_width = 32
        large_armor_width = 54

        roi_size = (20, 28)

        lights_vertices = np.array([
            self.left_light.btm, self.left_light.top,
            self.right_light.top, self.right_light.btm
        ]).astype(np.float32)

        top_light_y = (warp_height - light_length) / 2 - 1
        bottom_light_y = top_light_y + light_length

        if self.armor_type == 'small':
            warp_width = small_armor_width
        else:
            warp_width = large_armor_width

        target_vertices = np.array([
            [0, bottom_light_y], [0, top_light_y],
            [warp_width - 1, top_light_y], [warp_width - 1, bottom_light_y]
        ]).astype(np.float32)  # default to float64 for some reason...

        rotation_matrix = cv2.getPerspectiveTransform(
            lights_vertices, target_vertices)

        self.number_image = cv2.warpPerspective(
            rgb_img, rotation_matrix, (warp_width, warp_height))

        start_x = int((warp_width - roi_size[0]) / 2)
        self.number_image = self.number_image[:roi_size[1],
                                              start_x:start_x + roi_size[0]]

        # Binarize
        self.number_image = cv2.cvtColor(self.number_image, cv2.COLOR_RGB2GRAY)
        thres, self.number_image = cv2.threshold(
            self.number_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

class cv_armor_proposer:
    """Armor proposer using OpenCV."""

    # Hyperparameters
    MIN_LIGHTNESS = 160
    LUMINANCE_THRES = 90

    LIGHT_MIN_RATIO = 0.1
    LIGHT_MAX_RATIO = 0.55
    LIGHT_MAX_ANGLE = 40.0

    LIGHT_AREA_THRES = 50

    ARMOR_MIN_LIGHT_RATIO = 0.6
    ARMOR_MIN_SMALL_CENTER_DISTANCE = 1.3
    ARMOR_MAX_SMALL_CENTER_DISTANCE = 4
    ARMOR_MIN_LARGE_CENTER_DISTANCE = 4
    ARMOR_MAX_LARGE_CENTER_DISTANCE = 6

    assert ARMOR_MAX_SMALL_CENTER_DISTANCE <= ARMOR_MIN_LARGE_CENTER_DISTANCE

    armor_max_angle = 35.0

    def __init__(self, config, detect_color):
        """Initialize the armor proposer.

        Args:
            config (python object): shared config.py
            detect_color (int): color to detect (0: red, 1: blue ENUM)
        """
        self.CFG = config
        self.detect_color = detect_color

    def __call__(self, rgb_img):
        """Run the armor proposer for a single image.

        Args:
            rgb_img (np.ndarray): RGB image

        Returns:
            list of armor_class: list of detected armors
        """
        binary_img = self.preprocess(rgb_img)
        if self.CFG.DEBUG_DISPLAY:
            # visualize binary image
            cv2.imshow('binary', binary_img)
            cv2.waitKey(1)

        light_list = self.find_lights(rgb_img, binary_img)
        if self.CFG.DEBUG_DISPLAY:
            viz_img = rgb_img.copy()
            # visualize lights
            for light in light_list:
                cv2.rectangle(
                    viz_img, (int(
                        light.top[0]), int(
                        light.top[1])), (int(
                            light.btm[0]), int(
                            light.btm[1])), (0, 255, 0), 2)
            cv2.imshow('lights', viz_img)
            cv2.waitKey(1)

        armor_list = self.match_lights(light_list)
        if self.CFG.DEBUG_DISPLAY:
            viz_img = rgb_img.copy()
            # visualize armors
            for armor in armor_list:
                cv2.rectangle(
                    viz_img, (int(
                        armor.left_light.top[0]), int(
                        armor.left_light.top[1])), (int(
                            armor.right_light.btm[0]), int(
                            armor.right_light.btm[1])), (0, 255, 0), 2)
            cv2.imshow('armors', viz_img)
            cv2.waitKey(1)

        return armor_list

    def preprocess(self, rgb_img):
        """Preprocess for binarized image.

        Args:
            rgb_img (np.ndarray): RGB image

        Returns:
            np.ndarray: binarized image
        """
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        thres, binary_img = cv2.threshold(
            gray_img, self.MIN_LIGHTNESS, 255, cv2.THRESH_BINARY)

        return binary_img

    def filter_contours_rects(self, contours, rects, rgb_img):
        """Filter out contours that are too small and not of interested color.

        Args:
            contours (list of np.ndarray): contours
            rects (list of np.ndarray): bounding rectangles
            rgb_img (np.ndarray): RGB image

        Returns:
            list of np.ndarray: filtered contours
            list of np.ndarray: filtered bounding rectangles
        """
        filtered_contours = []
        filtered_rects = []
        assert len(contours) == len(rects)
        for i in range(len(rects)):
            x, y, w, h = rects[i]
            # Area test
            if w * h < self.LIGHT_AREA_THRES:
                continue

            # Color test
            if not Utils.color_test(rgb_img, rects[i], self.detect_color):
                continue

            filtered_rects.append(rects[i])
            filtered_contours.append(contours[i])

        return filtered_contours, filtered_rects

    def find_lights(self, rgb_img, binary_img):
        """Find potential light bars in the image.

        It follows open-sourced implementation to detect light bar from
        binarzied images. However, in practice we find that is not robust
        enough. So we augment it with color difference contours too.

        Two contours are often overlapped and we apply NMS to remove
        overlapped ones. We trust binary contours more than color difference.

        The NMS is also different from the one in object detection. Namely,
        in detection we usually do

            IoU = intersection / union

        So that a small object in front of a big object will not be removed.

        However, in our case, using this leads to inefficiency because two light bars
        can not be contained in each other. So we use

            IoU = intersection / area of smaller one

        instead.

        Args:
            rgb_img (np.ndarray): RGB image
            binary_img (np.ndarray): binarized image

        Returns:
            list of light_class: list of detected lights
        """
        bin_contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Use difference of color to handle white light
        if self.detect_color == RED:
            color_diff = rgb_img[:, :, 0].astype(int) - rgb_img[:, :, 2]
            color_diff[color_diff < 0] = 0
            color_diff = color_diff.astype(np.uint8)
            _, color_bin = cv2.threshold(
                color_diff, self.LUMINANCE_THRES, 255, cv2.THRESH_BINARY)
            color_contours, _ = cv2.findContours(
                color_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            color_diff = rgb_img[:, :, 2].astype(int) - rgb_img[:, :, 0]
            color_diff[color_diff < 0] = 0
            color_diff = color_diff.astype(np.uint8)
            _, color_bin = cv2.threshold(
                color_diff, self.LUMINANCE_THRES, 255, cv2.THRESH_BINARY)
            color_contours, _ = cv2.findContours(
                color_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.CFG.DEBUG_DISPLAY:
            cv2.imshow('color', color_bin)
            cv2.waitKey(1)

        # Preprocess contours for early filtering
        bin_contours = [
            contour for contour in bin_contours if contour.shape[0] >= 5]
        color_contours = [
            contour for contour in color_contours if contour.shape[0] >= 5]

        bin_rects = [cv2.boundingRect(contour) for contour in bin_contours]
        col_rects = [cv2.boundingRect(contour) for contour in color_contours]

        # Basic filtering for contours and rects
        bin_contours, bin_rects = self.filter_contours_rects(
            bin_contours, bin_rects, rgb_img)
        color_contours, col_rects = self.filter_contours_rects(
            color_contours, col_rects, rgb_img)

        # Apply NMS to contours from binary image and color difference image
        # COLOR contours are generally less reliable
        final_contours = []
        for col_rect, col_contour in zip(col_rects, color_contours):
            append_flag = True
            for bin_rect in bin_rects:
                # Compute rectangle overlap
                x_overlap = max(0,
                                min(bin_rect[0] + bin_rect[2],
                                    col_rect[0] + col_rect[2]) - max(bin_rect[0],
                                                                     col_rect[0]))
                y_overlap = max(0,
                                min(bin_rect[1] + bin_rect[3],
                                    col_rect[1] + col_rect[3]) - max(bin_rect[1],
                                                                     col_rect[1]))
                overlap = x_overlap * y_overlap
                min_area = min(bin_rect[2] * bin_rect[3],
                               col_rect[2] * col_rect[3])
                normalized_overlap = overlap / min_area
                NMS_THRES = 0.5
                if normalized_overlap > NMS_THRES:
                    # If overlap is too large, discard color contour
                    append_flag = False
                    break
            if append_flag:
                final_contours.append(col_contour)
        final_contours.extend(bin_contours)

        light_list = []
        for contour in final_contours:
            light = cv2.minAreaRect(contour)
            light = light_class(light)

            if not self.is_light(light):
                continue

            light.color = self.detect_color
            light_list.append(light)
        return light_list

    def is_light(self, light):
        """Apply filtering to determine if a light bar is valid.

        Criteria:
            1. Aspect ratio can not be too large or too small (light bars are slim verticals)
            2. Tilt angle (light bars are vertical)

        Args:
            light (light_class): light bar to be filtered

        Returns:
            bool: True if the light bar is valid
        """
        ratio = light.width / light.length
        ratio_ok = self.LIGHT_MIN_RATIO < ratio and ratio < self.LIGHT_MAX_RATIO

        angle_ok = light.tilt_angle < self.LIGHT_MAX_ANGLE

        return ratio_ok and angle_ok

    def match_lights(self, light_list):
        """Match pairs of lights into armors.

        Args:
            light_list (list of light_class): list of detected lights

        Returns:
            list of armor_class: list of detected armors
        """
        armor_list = []

        for i in range(len(light_list)):
            for j in range(i + 1, len(light_list)):
                light1 = light_list[i]
                light2 = light_list[j]

                assert light1.color == self.detect_color
                assert light2.color == self.detect_color

                if self.contain_light(light1, light2, light_list):
                    continue

                armor = armor_class(light1, light2, self.detect_color)

                if self.is_armor(armor):
                    armor_list.append(armor)

        return armor_list

    def contain_light(self, light1, light2, light_list):
        """Test if a pair of light is contained in another light bar.

        Empirically, this is a good way to filter out false positives.

        Args:
            light1 (light_class): light bar 1
            light2 (light_class): light bar 2
            light_list (list of light_class): list of all detected lights

        Returns:
            bool: True if light1 and light2 are contained in another light bar
        """
        pts = np.array([
            light1.top, light1.btm, light2.top, light2.btm
        ])
        rect = cv2.boundingRect(pts)

        for test_light in light_list:
            if test_light == light1 or test_light == light2:
                continue

            if Utils.rect_contains(
                    rect, test_light.top) or Utils.rect_contains(
                    rect, test_light.btm) or Utils.rect_contains(
                    rect, test_light.center):
                return True

        return False

    def is_armor(self, armor):
        """Apply filtering to determine if an armor is valid.

        Criteria:
            1. The length of two light bars can not be too different
            2. The ratio of distance between two light bars and their length
            3. The angle of armor boards two light bars would form

        Args:
            armor (armor_class): armor to be filtered

        Returns:
            bool: True if the armor meets all the criteria
        """
        light1 = armor.left_light
        light2 = armor.right_light
        if light1.length < light2.length:
            light_length_ratio = light1.length / light2.length
        else:
            light_length_ratio = light2.length / light1.length

        light_ratio_ok = light_length_ratio > self.ARMOR_MIN_LIGHT_RATIO

        avg_light_length = (light1.length + light2.length) / 2
        center_dist = cv2.norm(
            light1.center - light2.center) / avg_light_length

        center_dist_ok = (
            (self.ARMOR_MIN_SMALL_CENTER_DISTANCE < center_dist) and (
                center_dist < self.ARMOR_MAX_SMALL_CENTER_DISTANCE)) or (
            (self.ARMOR_MIN_LARGE_CENTER_DISTANCE < center_dist) and (
                center_dist < self.ARMOR_MAX_LARGE_CENTER_DISTANCE))

        # test light center connection angle
        diff = light1.center - light2.center
        angle = abs(np.arctan(diff[1] / diff[0])) / np.pi * 180
        angle_ok = angle < self.armor_max_angle

        if center_dist > self.ARMOR_MIN_LARGE_CENTER_DISTANCE:
            armor.armor_type = 'large'
        else:
            armor.armor_type = 'small'

        return light_ratio_ok and center_dist_ok and angle_ok

class dl_digit_classifier:
    """Classify digits in armor number using deep learning model."""

    LABEL_NAMES_LIST = np.array(['B', '1', '2', '3', '4', '5', 'G', 'O', 'N'])
    CLASSIFIER_THRESHOLD = 0.7

    def __init__(self, config, model_path):
        """Initialize the classifier.

        Args:
            config (python object): shared config.py
            model_path (str): path to the model file
        """
        self.CFG = config
        self.net = cv2.dnn.readNetFromONNX(model_path)

    @staticmethod
    def normalize(image):
        """Normalize image to [0, 1]."""
        return image.astype(np.float32) / 255.0

    def __call__(self, armor_list):
        """Classify a batch of armor numbers.

        Args:
            armor_list (list of armor_class): list of armors to be classified

        Returns:
            list of str: list of classified armor numbers
        """
        ret_list = []
        for armor in armor_list:
            input_image = self.normalize(armor.number_image)
            blob = cv2.dnn.blobFromImage(
                input_image, scalefactor=1., size=(28, 20))  # BCHW
            self.net.setInput(blob)
            outputs = self.net.forward()
            max_probs = outputs.max(axis=1)
            softmax_probs = np.exp(outputs - max_probs)
            my_sum = np.sum(softmax_probs, axis=1)
            softmax_probs = softmax_probs / my_sum

            # Test
            max_class = softmax_probs.argmax(axis=1)
            max_class_names = self.LABEL_NAMES_LIST[max_class]

            if softmax_probs[0,
                             max_class] < self.CLASSIFIER_THRESHOLD or max_class_names == 'N':
                continue

            # TODO: use digit predictions to improve accuracy?
            # Right now using that logic causes a lot of false negatives...
            armor.confidence = softmax_probs[0, max_class]
            ret_list.append(armor)

        return ret_list
