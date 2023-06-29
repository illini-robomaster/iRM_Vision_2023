"""Wrap the MDVS SDK to build a camera driver."""
import cv2
import numpy as np
import time
import Utils

from Camera import mvsdk
from Camera.camera_base import CameraBase

class mdvs_camera(CameraBase):
    """MDVS camera driver.

    Adapted from official MDVS SDK example code, which is located at
    https://mindvision.com.cn/uploadfiles/2021/04/20/75287253862433139.zip

    More official MDVS demo programs can be found here
        https://mindvision.com.cn/rjxz/list_12.aspx?lcid=139
    """

    # Computed using the tool at https://mindvision.com.cn/jtxx/list_108.aspx?lcid=21&lcids=1656
    # Config: 6mm lens, MV-SUA133GC
    YAW_FOV_HALF = Utils.deg_to_rad(46.245) / 2
    PITCH_FOV_HALF = Utils.deg_to_rad(37.761) / 2

    def __init__(self, cfg, camera_event=None):
        """Initialize the MDVS camera.

        Args:
            cfg (python object): shared config object
        """
        super().__init__(cfg)

        # self.last_time = time.time()
        # self.print_last_time = time.time()
        # self.fps_txt = 0
        self.event_time = -1
        self.event = camera_event
        self.frame = None

        # Enumerate camera devices
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print(
                "{}: {} {}".format(
                    i,
                    DevInfo.GetFriendlyName(),
                    DevInfo.GetPortType()))
        i = 0 if nDev == 1 else int(input("Select camera: "))
        DevInfo = DevList[i]
        print(DevInfo)

        # Initialize connection to camera
        self.cam = 0
        try:
            self.cam = mvsdk.CameraInit(DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return

        # Get camera capability (tech specs)
        cap = mvsdk.CameraGetCapability(self.cam)

        # Test if the camera is a mono camera or color camera
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # Set camera output format based on mono or color
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(
                self.cam, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.cam, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # Set camera trigger mode
        mvsdk.CameraSetTriggerMode(self.cam,
                                   self.cfg.TRIGGER_MODE)  # 0: continuous acquisition mode; 1: software trigger
        # mode; 2: hardware trigger mode

        # Set to manual exposure mode and set exposure time to 30ms
        mvsdk.CameraSetAeState(self.cam, 0)
        mvsdk.CameraSetExposureTime(self.cam, self.exposure_time * 1000)

        # Calls SDK internal thread to start grabbing images
        # If in trigger mode, the image grabbing won't start until a trigger
        # frame is received
        mvsdk.CameraPlay(self.cam)

        # Compute how much buffer is needed to store the image
        # Use maximum supported resolution of the camera to compute
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # Allocate RGB buffer, used to store the image converted from RAW by ISP
        # Remark: RAW data is transferred to PC, and then converted to RGB by software ISP
        #         If it is a monochrome camera, the format does not need to be converted
        # but the ISP still has other processing, so this buffer is still
        # needed
        self.frame_buffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

        self.quit = False
        mvsdk.CameraSetCallbackFunction(self.cam, self.grab_callback, 0)

        while not self.quit:
            time.sleep(0.1)

    def get_frame(self):
        """Call to get a frame from the camera.

        Raises:
            Exception: raised when video file is exhausted

        Returns:
            np.ndarray: RGB image frame
        """
        # Grab one frame from camera
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.cam, self.cfg.GRAB_WAIT)
            mvsdk.CameraImageProcess(self.cam, pRawData, self.frame_buffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.cam, pRawData)

            # At this point the frame is stored in pFrameBuffer.
            # For color cameras, pFrameBuffer = RGB data, and formono cameras,
            # pFrameBuffer = 8-bit grayscale data
            frame_data = (
                    mvsdk.c_ubyte *
                    FrameHead.uBytes).from_address(
                self.frame_buffer)

            # Convert C array to numpy array
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape(
                (FrameHead.iHeight,
                 FrameHead.iWidth,
                 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if self.cfg.ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            return frame
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

    @mvsdk.method(mvsdk.CAMERA_SNAP_PROC)
    def grab_callback(self, hCamera, pRawData, pFrameHead, pContext):
        # elasped_time = time.time() - self.last_time
        # self.last_time = time.time()
        # fps = 1.0 / elasped_time

        FrameHead = pFrameHead[0]

        mvsdk.CameraImageProcess(hCamera, pRawData, self.frame_buffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.frame_buffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                               1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if self.cfg.ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        self.frame = frame
        self.event_time = time.time()
        self.event.set()

        # if time.time() - self.print_last_time > 0.1:
        #     self.fps_txt = fps
        #     self.print_last_time = time.time()
        # frame = cv2.putText(frame, "FPS: {:.1f}".format(self.fps_txt), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
        #                     cv2.LINE_AA)
        #
        # cv2.imshow("Press q to end", frame)
        # if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #     self.quit = True

    def get_event_time(self):
        return self.event_time

    def get_frame_cache(self):
        return self.frame

    def __del__(self):
        """Clean up MDVS driver connection and buffer."""
        # Shut down SDK camera internal thread
        mvsdk.CameraUnInit(self.cam)

        # Release frame buffer
        mvsdk.CameraAlignFree(self.frame_buffer)
