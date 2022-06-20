# Commonly used utils functions
import numpy as np

def deg_to_rad(deg):
    return deg * ((2 * np.pi) / 360)

def rad_to_deg(rad):
    return rad * (360. / (2 * np.pi))

def estimate_target_depth(bbox, depth_map):
    center_x, center_y, width, height = bbox

    upper_left_x = int(center_x - width / 2)
    upper_left_y = int(center_y - height / 2)
    lower_right_x = int(center_x + width / 2)
    lower_right_y = int(center_y + height / 2)

    # Get distance to target
    depth_region = depth_map[upper_left_y:lower_right_y,upper_left_x:lower_right_x]
    estimated_depth = np.mean(depth_region[depth_region > 0])

    if np.isnan(estimated_depth):
        # if NaN (i.e., all depth observations are invalid), use bbox size for a rough estimation
        # intentionally scale to be 'virtually' further than estimated depth
        # so that
        #       1. samples with valid depth estimation are preferred
        #       2. samples with larger bbox are preferred
        estimated_depth = 100000 + (100. / (width * height))
    else:
        assert estimated_depth >= 0 and estimated_depth <= 2**16 - 1

    return estimated_depth
