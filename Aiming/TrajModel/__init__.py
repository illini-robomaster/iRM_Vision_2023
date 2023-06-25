"""Aggregate all the trajectory modeling methods into one module.

For a complete explanation of each module, please refer to the README.md.
"""

from .gravity import calibrate_pitch_gravity
from .barrel_camera import get_camera_barrel_T
from .barrel_to_robot import barrel_to_robot_T
