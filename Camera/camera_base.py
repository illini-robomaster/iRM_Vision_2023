"""Base class for all camera implementations."""
from abc import abstractmethod

class CameraBase:
    """Base class for all camera implementations."""

    def __init__(self, cfg):
        """Initialize the camera.

        Args:
            cfg (python object): shared config object
        """
        self.cfg = cfg
        self.width = self.cfg.IMG_WIDTH
        self.height = self.cfg.IMG_HEIGHT
        self.exposure_time = int(self.cfg.EXPOSURE_TIME)

    @abstractmethod
    def get_frame(self):
        """Call to get a frame from the camera.

        Returns:
            np.ndarray: RGB image frame
        """
        raise NotImplementedError
