"""YOLO detector using ONNX network."""
import os
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision

try:
    import onnxruntime
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from .onnx_tensorrt import TensorRTBackend
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

if not HAS_ORT and not HAS_TRT:
    raise ImportError(
        'You need to install either onnxruntime or tensorrt to use YOLOv7 detector')

# Implementation based on YOLOv7


def xywh2xyxy_export(cx, cy, w, h):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2].

    xy1=top-left, xy2=bottom-right

    Args:
        cx (float): center x
        cy (float): center y
        w (float): width
        h (float): height

    Returns:
        torch.tensor: converted boxes
    """
    # This function is used while exporting ONNX models
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    halfw = w / 2
    halfh = h / 2
    xmin = cx - halfw  # top left x
    ymin = cy - halfh  # top left y
    xmax = cx + halfw  # bottom right x
    ymax = cy + halfh  # bottom right y
    return torch.cat((xmin, ymin, xmax, ymax), 1)


def non_max_suppression_export(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        nc=None):
    """Run Non-Maximum Suppression (NMS) on inference results.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if nc is None:
        nc = prediction.shape[2] - 5

    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    xc = prediction[..., 4] > conf_thres  # candidates
    output = []
    for _ in range(prediction.shape[0]):
        output.append(torch.zeros((0, 5 + nc), device=prediction.device))
    for xi, x in enumerate(prediction):  # image index, image inference
        idx = xc[0, :]
        x = x[idx, :]  # confidence
        # Compute conf
        cx, cy, w, h = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        obj_conf = x[:, 4:5]
        cls_conf = x[:, 5:5 + nc]
        cls_conf = obj_conf * cls_conf  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy_export(cx, cy, w, h)
        conf, j = cls_conf.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)
        x = x[conf.view(-1) > conf_thres]
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        output[xi] = x[i]
    return output


class yolo_detector:
    """YOLO detector using ONNX backbone."""

    CLASS_NAMES_LIST = [
        'B1',
        'B2',
        'B3',
        'B4',
        'B5',
        'BO',
        'BS',
        'R1',
        'R2',
        'R3',
        'R4',
        'R5',
        'RO',
        'RS']

    def __init__(self, cfg):
        """Initialize the YOLO detector.

        Args:
            cfg (python object): python config node object
        """
        self.CFG = cfg
        assert os.path.exists(self.CFG.YOLO_PATH), self.CFG.YOLO_PATH

        # ===== Populate preprocessing params =====
        self.H, self.W = self.CFG.IMG_HEIGHT, self.CFG.IMG_WIDTH

        target_shape = max(self.H, self.W), max(self.H, self.W)
        self.target_H, self.target_W = target_shape

        # TODO(roger): input image dimension is frozen due to frozen static ONNX
        assert self.H == 512
        assert self.W == 640

        assert self.target_H == 640
        assert self.target_W == 640

        self.dh = self.target_H - self.H
        self.half_dh = self.dh / 2

        self.padding_top, self.padding_btm = int(
            round(self.half_dh - 0.1)), int(round(self.half_dh + 0.1))
        self.padding_left, self.padding_right = 0, 0

        # Prep inference backend
        if HAS_TRT:  # TRT is generally faster than ORT. Thus higher priority.
            if self.CFG.INT8_QUANTIZATION:
                self.int8_calibrator = self._prep_int8_calibrator()
            else:
                self.int8_calibrator = None
            serialized_engine_path = self.CFG.YOLO_PATH + '.trt'
            self.trt_engine = TensorRTBackend.prepare(self.CFG.YOLO_PATH,
                                                      device='CUDA:0',
                                                      serialize_engine=True,
                                                      verbose=False,
                                                      serialized_engine_path=serialized_engine_path,
                                                      int8_calibrator=self.int8_calibrator)
        elif HAS_ORT:
            if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                providers = ['CUDAExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(self.CFG.YOLO_PATH, providers=providers)
        else:
            raise NotImplementedError

    def detect(self, rgb_img):
        """Detect armor in the input image.

        Args:
            rgb_img (np.array): input image in RGB format

        Returns:
            list: list of prediction tuples
        """
        assert rgb_img.shape[:2] == (512, 640)
        img = self.pad_img(rgb_img)

        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # to CHW

        img = img / 255.0
        img = img[None].astype(np.float32)

        if HAS_TRT:
            outputs = self.trt_engine.run(img)
            raw_pred = outputs[3]
        elif HAS_ORT:
            outputs = self.session.run(None, {"images": img})
            raw_pred = outputs[0]
        else:
            raise NotImplementedError

        bbox_list = non_max_suppression_export(torch.tensor(raw_pred))
        assert len(bbox_list) == 1, 'input BS is 1'

        ret_list = []

        for pred_idx in range(bbox_list[0].shape[0]):
            min_x, min_y, max_x, max_y, conf, armor_type = self.unpad_pred(bbox_list[0][pred_idx])
            min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
            conf = float(conf)
            cls_name = self.CLASS_NAMES_LIST[int(armor_type)]

            ret_tuple = (min_x, min_y, max_x, max_y, conf, cls_name)

            ret_list.append(ret_tuple)

        return ret_list

    def pad_img(self, img):
        """Pad the input image to the target shape.

        Args:
            img (np.array): input image in RGB format

        Returns:
            np.array: padded image
        """
        # To generalize this function, see letterbox function in YOLOv7
        assert img.shape[0] == self.H
        assert img.shape[1] == self.W

        DEFAULT_COLOR = (114, 114, 114)  # from YOLO official

        img = cv2.copyMakeBorder(img,
                                 self.padding_top,
                                 self.padding_btm,
                                 self.padding_left,
                                 self.padding_right,
                                 cv2.BORDER_CONSTANT,
                                 value=DEFAULT_COLOR)

        return img

    def unpad_pred(self, pred_result):
        """Unpad the prediction result to the original image shape.

        Args:
            pred_result (tuple): a single prediction tuple

        Returns:
            tuple: pred result in the same format, with offseted bbox removing padding
        """
        min_x, min_y, max_x, max_y, conf, cls = pred_result
        min_y -= self.padding_top
        max_y -= self.padding_top
        min_x -= self.padding_left
        max_x -= self.padding_left
        return min_x, min_y, max_x, max_y, conf, cls

    def _prep_int8_calibrator(self):
        # TODO: make this function flexible by supporting net_hw params
        def prep_yolo_img_from_path(img_path, net_hw):
            rgb_img = Image.open(img_path).convert('RGB')
            rgb_img_np = np.array(rgb_img)

            # ONNX height and width are 640
            # Step 1: Resize longer edge to 640
            H, W = rgb_img_np.shape[:2]

            if H > W:
                target_H = 640
                target_W = int((target_H / H) * W)
            else:
                target_W = 640
                target_H = int((target_W / W) * H)
            
            rgb_img = rgb_img.resize((target_W, target_H))
            rgb_img_np = np.array(rgb_img)

            # Step 2: Pad shorter edge to 640
            H, W = rgb_img_np.shape[:2]

            dh = 640 - H
            half_dh = dh / 2

            padding_top, padding_btm = int(round(half_dh - 0.1)), int(round(half_dh + 0.1))

            dw = 640 - W
            half_dw = dw / 2

            padding_left, padding_right = int(round(half_dw - 0.1)), int(round(half_dw + 0.1))

            DEFAULT_COLOR = (114, 114, 114)  # from YOLO official

            rgb_img_np = cv2.copyMakeBorder(rgb_img_np,
                                    padding_top,
                                    padding_btm,
                                    padding_left,
                                    padding_right,
                                    cv2.BORDER_CONSTANT,
                                    value=DEFAULT_COLOR)

            # Step 3: normalization
            assert rgb_img_np.shape[:2] == (640, 640)

            rgb_img_np = np.ascontiguousarray(rgb_img_np.transpose(2, 0, 1))  # to CHW

            rgb_img_np = rgb_img_np.astype(np.float32) / 255.0

            return rgb_img_np
    
        from .int8_calibrator import int8_calibrator

        my_calibrator = int8_calibrator(
            self.CFG.CALIB_IMG_DIR,
            (640, 640),
            self.CFG.YOLO_PATH + '.int8_calib_cache.bin',
            prep_yolo_img_from_path,
            1,
        )

        return my_calibrator
