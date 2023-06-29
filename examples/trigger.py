import time

import cv2
from Camera.mdvs import mdvs_camera
import config


def main():
    camera = mdvs_camera(config)


if __name__ == "__main__":
    main()
