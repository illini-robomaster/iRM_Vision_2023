import os
import numpy as np

class basic_tracker(object):
    '''
    Basic tracker that can handle only one target.

    It memorizes the state of last two predictions and do linear extrapolation
    '''
    def __init__(self):
        self.last_pred = None
        self.last_last_pred = None

    def register_one(self, pred_list, enemy_team, rgb_img):
        # TODO: add a tracking flag in pred_list to distinguish
        # tracked and detected objects
        my_pred_list = [i for i in pred_list if i[0] == f"armor_{enemy_team}"]

        assert len(my_pred_list) <= 1, "Basic tracker does not support multiple targets"

        self.last_last_pred = self.last_pred
        self.last_pred = my_pred_list

    def fix_prediction(self, cur_pred_list):
        if len(cur_pred_list) == 0:
            last_name, last_conf, last_bbox = self.last_pred[0]
            last_last_name, last_last_conf, last_last_bbox = self.last_last_pred[0]

            assert last_name == last_last_name

            last_center_x, last_center_y, last_width, last_height = last_bbox
            last_last_center_x, last_last_center_y, last_last_width, last_last_height = last_last_bbox

            # Linear extrapolation
            center_x = last_center_x + (last_center_x - last_last_center_x) / 2
            center_y = last_center_y + (last_center_y - last_last_center_y) / 2
            width = last_width + (last_width - last_last_width) / 2
            height = last_height + (last_height - last_last_height) / 2

            return [[last_name, (last_conf + last_last_conf) / 2, [center_x, center_y, width, height]]]
        else:
            # If not empty, do nothing
            return cur_pred_list
