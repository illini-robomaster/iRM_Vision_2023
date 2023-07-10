"""Generic INT8 calibrator modified from https://github.com/jkjung-avt/tensorrt_demos ."""

import os
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver


class int8_calibrator(trt.IInt8EntropyCalibrator2):
    """A generic INT8 calibratior for TRT.

    This class implements TensorRT's IInt8EntropyCalibtrator2 interface.
    It reads all images from the specified directory and generates INT8
    calibration data for accordingly.
    """

    def __init__(self, img_dir, net_hw, cache_file, img_read_func, batch_size=1):
        """Initialize INT8 calibrator.

        Args:
            img_dir (str): path to dir containing calibration images in `.jpg`.
            net_hw (tuple): (height, width) to which the calib images will be processed
            cache_file (str): path to cache file
            img_read_func (func): a func takes in a path and size to return np.array image.
            batch_size (int, optional): Batch size during calibration. Defaults to 1.

        Raises:
            FileNotFoundError: when the calib img dir does not exist
        """
        if not os.path.isdir(img_dir):
            raise FileNotFoundError('%s does not exist' % img_dir)

        super().__init__()  # trt.IInt8EntropyCalibrator2.__init__(self)

        self.img_dir = img_dir
        self.net_hw = net_hw
        self.cache_file = cache_file
        self.img_read_func = img_read_func
        self.batch_size = batch_size
        self.blob_size = 3 * net_hw[0] * net_hw[1] * np.dtype('float32').itemsize * batch_size

        self.jpgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        # The number "500" is NVIDIA's suggestion.  See here:
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimizing_int8_c
        if len(self.jpgs) < 500:
            print('WARNING: found less than 500 images in %s!' % img_dir)
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = pycuda.driver.mem_alloc(self.blob_size)

    def __del__(self):
        """Define a destructor that frees allocated CUDA memory."""
        del self.device_input

    def get_batch_size(self):
        """Return batch size for calibration. Required by TRT."""
        return self.batch_size

    def get_batch(self, names):
        """Get a batch of images. Names is an optional param used to define order."""
        if self.current_index + self.batch_size > len(self.jpgs):
            return None

        batch = []
        for i in range(self.batch_size):
            img_path = os.path.join(
                self.img_dir, self.jpgs[self.current_index + i])
            batch.append(self.img_read_func(img_path, self.net_hw))
        batch = np.stack(batch)
        assert batch.nbytes == self.blob_size

        pycuda.driver.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        """Read calibration results from a cache file.

        If there is a cache, use it instead of calibrating again.
        Otherwise, implicitly return None.

        Function signature and return format defined by TRT here:
        https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Int8/EntropyCalibrator2.html
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        """Write calib result to a cache file.

        Defined by TRT in the same API docs in the above link in self.read_calibration_cache.
        """
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
