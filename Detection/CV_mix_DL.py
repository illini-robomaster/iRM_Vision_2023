import numpy as np
import cv2
import os
import time

def auto_align_brightness(img, target_v=50):
    # Only decrease!
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

class cv_mix_dl_detector:
    def __init__(self, detect_color, model_path='fc.onnx'):
        self.armor_proposer = cv_armor_proposer(detect_color)
        self.digit_classifier = dl_digit_classifier(model_path)
        self.change_color(detect_color)
    
    def detect(self, rgb_img):
        return self(rgb_img)
    
    def __call__(self, rgb_img):
        rgb_img = auto_align_brightness(rgb_img)
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
            width = cv2.norm(armor.left_light.center - armor.right_light.center)
            bbox = (center_x, center_y, width, height)
            name = f'armor_{self.detect_color_str}'
            conf = armor.confidence
            ret_list.append((name, conf, bbox))
        return ret_list

    def change_color(self, new_color):
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

def rect_contains(rect,pt):
    return rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]

class light_class:
    def __init__(self, rotated_rect):
        ((self.center_x, self.center_y), (_, _), _) = rotated_rect
        # sort according to y
        pts = sorted(cv2.boxPoints(rotated_rect), key = lambda x : x[1])
        self.top = (pts[0] + pts[1]) / 2
        self.btm = (pts[2] + pts[3]) / 2

        self.length = cv2.norm(self.top - self.btm)
        self.width = cv2.norm(pts[0] - pts[1])

        self.tilt_angle = np.arctan2(np.abs(self.top[0] - self.btm[0]), np.abs(self.top[1] - self.btm[1]))
        self.tilt_angle = self.tilt_angle / np.pi * 180
        
        self.center = np.array([self.center_x, self.center_y])
        
        self.color = None

class armor_class:
    def __init__(self, light1, light2, color):
        assert light1.color == light2.color
        assert light1.color == color
        self.color = color
        
        if light1.center_x < light2.center_x:
            self.left_light = light1
            self.right_light = light2
        else:
            self.left_light = light2
            self.right_light = light1
        
        self.center = (self.left_light.center + self.right_light.center) / 2
        self.armor_type = None # 'large', 'small'
    
    def extract_number(self, rgb_img):
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
        ]).astype(np.float32) # default to float64 for some reason...
        
        rotation_matrix = cv2.getPerspectiveTransform(lights_vertices, target_vertices)
        
        self.number_image = cv2.warpPerspective(rgb_img, rotation_matrix, (warp_width, warp_height))
        
        start_x = int((warp_width - roi_size[0]) / 2)
        self.number_image = self.number_image[:roi_size[1], start_x:start_x+roi_size[0]]
        
        # Binarize
        self.number_image = cv2.cvtColor(self.number_image, cv2.COLOR_RGB2GRAY)
        thres, self.number_image = cv2.threshold(self.number_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

class cv_armor_proposer:
    # Hyperparameters
    MIN_LIGHTNESS = 160

    LIGHT_MIN_RATIO = 0.1
    LIGHT_MAX_RATIO = 0.55
    LIGHT_MAX_ANGLE = 40.0

    ARMOR_MIN_LIGHT_RATIO = 0.6
    ARMOR_MIN_SMALL_CENTER_DISTANCE = 1.3
    ARMOR_MAX_SMALL_CENTER_DISTANCE = 4
    ARMOR_MIN_LARGE_CENTER_DISTANCE = 4
    ARMOR_MAX_LARGE_CENTER_DISTANCE = 6

    assert ARMOR_MAX_SMALL_CENTER_DISTANCE <= ARMOR_MIN_LARGE_CENTER_DISTANCE

    armor_max_angle = 35.0

    def __init__(self, detect_color):
        self.detect_color = detect_color
    
    def __call__(self, rgb_img):
        binary_img = self.preprocess(rgb_img)
        light_list = self.find_lights(rgb_img, binary_img)
        armor_list = self.match_lights(light_list)
        
        return armor_list
    
    def preprocess(self, rgb_img):
        """
        Return binarized image
        """
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

        thres, binary_img = cv2.threshold(gray_img, self.MIN_LIGHTNESS, 255, cv2.THRESH_BINARY)

        return binary_img

    def find_lights(self, rgb_img, binary_img):
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        light_list = []
        start_cp = time.time()
        for contour in contours:
            if contour.shape[0] < 5:
                continue

            light = cv2.minAreaRect(contour)
            light = light_class(light)
            if not self.is_light(light):
                continue

            bounding_rect = cv2.boundingRect(contour)

            x, y, w, h = bounding_rect

            if 0 <= x and 0 <= w and (x + w) <= rgb_img.shape[1] and 0 < y and 0 < h and y + h <=rgb_img.shape[0]:
                sum_r = 0
                sum_b = 0

                roi = rgb_img[y:y+h,x:x+w]

                for i in range(roi.shape[0]):
                    for j in range(roi.shape[1]):
                        if cv2.pointPolygonTest(contour, (j + x, i + y), False):
                            # point is inside contour
                            sum_r += roi[i,j,0]
                            sum_b += roi[i,j,2]

                if sum_r > sum_b:
                    my_color = 0 # RED
                else:
                    my_color = 1 # BLUE
                light.color = my_color
                light_list.append(light)
        return light_list
    
    def is_light(self, light):
        ratio = light.width / light.length
        ratio_ok = self.LIGHT_MIN_RATIO < ratio and ratio < self.LIGHT_MAX_RATIO

        angle_ok = light.tilt_angle < self.LIGHT_MAX_ANGLE
        
        return ratio_ok and angle_ok
    
    def match_lights(self, light_list):
        armor_list = []

        for i in range(len(light_list)):
            for j in range(i + 1, len(light_list)):
                light1 = light_list[i]
                light2 = light_list[j]

                if light1.color != self.detect_color or light2.color != self.detect_color:
                    continue

                if self.contain_light(light1, light2, light_list):
                    continue

                armor = armor_class(light1, light2, self.detect_color)

                if self.is_armor(armor):
                    armor_list.append(armor)
        
        return armor_list
    
    def contain_light(self, light1, light2, light_list):
        pts = np.array([
            light1.top, light1.btm, light2.top, light2.btm
        ])
        rect = cv2.boundingRect(pts)
        
        for test_light in light_list:
            if test_light == light1 or test_light == light2:
                continue
            
            if rect_contains(rect, test_light.top) or rect_contains(rect, test_light.btm)\
                        or rect_contains(rect, test_light.center):
                return True
        
        return False
    
    def is_armor(self, armor):
        light1 = armor.left_light
        light2 = armor.right_light
        if light1.length < light2.length:
            light_length_ratio = light1.length / light2.length
        else:
            light_length_ratio = light2.length / light1.length
        
        light_ratio_ok = light_length_ratio > self.ARMOR_MIN_LIGHT_RATIO
        
        avg_light_length = (light1.length + light2.length) / 2
        center_dist = cv2.norm(light1.center - light2.center) / avg_light_length
        
        center_dist_ok = ((self.ARMOR_MIN_SMALL_CENTER_DISTANCE < center_dist)\
                        and (center_dist < self.ARMOR_MAX_SMALL_CENTER_DISTANCE)) or\
                        ((self.ARMOR_MIN_LARGE_CENTER_DISTANCE < center_dist)\
                        and (center_dist < self.ARMOR_MAX_LARGE_CENTER_DISTANCE))
        
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
    LABEL_NAMES_LIST = np.array(['B', '1', '2', '3', '4', '5', 'G', 'O', 'N'])
    CLASSIFIER_THRESHOLD = 0.7

    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(model_path)
    
    @staticmethod
    def normalize(image):
        return image.astype(np.float32) / 255.0
    
    def __call__(self, armor_list):
        ret_list = []
        for armor in armor_list:
            input_image = self.normalize(armor.number_image)
            blob = cv2.dnn.blobFromImage(input_image, scalefactor=1., size=(28, 20)) # BCHW
            self.net.setInput(blob)
            outputs = self.net.forward()
            max_probs = outputs.max(axis = 1)
            softmax_probs = np.exp(outputs - max_probs)
            my_sum = np.sum(softmax_probs, axis=1)
            softmax_probs = softmax_probs / my_sum
            
            # Test
            max_class = softmax_probs.argmax(axis=1)
            max_class_names = self.LABEL_NAMES_LIST[max_class]
            
            if softmax_probs[0, max_class] < self.CLASSIFIER_THRESHOLD or max_class_names == 'N':
                continue
            
            # TODO: use digit predictions to improve accuracy?
            # Right now using that logic causes a lot of false negatives...
            armor.confidence = softmax_probs[0, max_class]
            ret_list.append(armor)

        return ret_list
