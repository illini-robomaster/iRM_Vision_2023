import time
import cv2
from Aiming.Aim import Aim
from Communication.communicator import serial_port, create_packet
# from Detection.YOLO import Yolo
from Detection.CV_mix_DL import cv_mix_dl_detector
import config

DEFAULT_ENEMY_TEAM = 'red'

class serial_circular_buffer:
    def __init__(self, buffer_size=10):
        self.buffer = []
        self.buffer_size = buffer_size
        self.default_color = DEFAULT_ENEMY_TEAM
    
    def receive(self, byte_array):
        for c in byte_array:
            if len(self.buffer) >= self.buffer_size:
                self.buffer = self.buffer[1:] # pop first element
            self.buffer.append(c)

    def get_enemy_color(self):
        # TODO: if a robot is revived, the serial port might get
        # garbage value in between...
        blue_cnt = 0
        red_cnt = 0

        for l in self.buffer:
            if l == ord('R'): red_cnt += 1
            if l == ord('B'): blue_cnt += 1
        
        if blue_cnt > red_cnt:
            self.default_color = 'blue'
            return 'blue'
        
        if red_cnt > blue_cnt:
            self.default_color = 'red'
            return 'red'
        
        return self.default_color

if __name__ == "__main__":
    model = cv_mix_dl_detector(config, DEFAULT_ENEMY_TEAM)
    # model = Yolo(config.MODEL_CFG_PATH, config.WEIGHT_PATH, config.META_PATH)
    aimer = Aim()
    communicator = serial_port
    pkt_seq = 0

    autoaim_camera = config.AUTOAIM_CAMERA(config.IMG_WIDTH, config.IMG_HEIGHT)

    # color buffer which retrieves enemy color from STM32
    my_color_buffer = serial_circular_buffer()

    if communicator is None:
        print("SERIAL DEVICE IS NOT AVAILABLE!!!")

    while True:
        start = time.time()
        frame = autoaim_camera.get_frame()

        if communicator is not None:
            if (communicator.inWaiting() > 0):
                # read the bytes and convert from binary array to ASCII
                byte_array = communicator.read(communicator.inWaiting())
                my_color_buffer.receive(byte_array)
        
        enemy_team = my_color_buffer.get_enemy_color()
        model.change_color(enemy_team)

        pred = model.detect(frame)

        for i in range(len(pred)):
            name, conf, bbox = pred[i]
            # name from C++ string is in bytes; decoding is needed
            if isinstance(name, bytes):
                name_str = name.decode('utf-8')
            else:
                name_str = name
            pred[i] = (name_str, conf, bbox)

        elapsed = time.time()-start

        if config.DEBUG_DISPLAY:
            viz_frame = frame.copy()
            for name, conf, bbox in pred:
                lower_x = int(bbox[0] - bbox[2] / 2)
                lower_y = int(bbox[1] - bbox[3] / 2)
                upper_x = int(bbox[0] + bbox[2] / 2)
                upper_y = int(bbox[1] + bbox[3] / 2)
                viz_frame = cv2.rectangle(viz_frame, (lower_x, lower_y), (upper_x, upper_y), (0, 255, 0), 2)
            cv2.imshow('all_detected', viz_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)
        
        # Tracking and filtering
        ret_dict = aimer.process_one(pred, enemy_team, frame)

        show_frame = frame.copy()

        if ret_dict:
            packet = create_packet(config.MOVE_YOKE, pkt_seq, ret_dict['yaw_diff'], ret_dict['pitch_diff'])
            show_frame = cv2.circle(show_frame,
                                    (int(ret_dict['center_x']), int(ret_dict['center_y'])),
                                    10, (0, 255, 0), 10)
        else:
            show_frame = cv2.putText(show_frame, 'NOT FOUND', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            packet = create_packet(config.SEARCH_TARGET, pkt_seq, 0, 0)g
        
        if config.DEBUG_DISPLAY:
            print('----------------\n',pred)
            print('fps:',1./elapsed)
            cv2.imshow('target', show_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)

        if communicator is not None:
            communicator.write(packet)

        pkt_seq += 1
