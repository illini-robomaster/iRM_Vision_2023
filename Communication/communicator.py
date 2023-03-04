import os
import serial
import crc
import threading

class UARTCommunicator(object):
    def __init__(self, cfg, crc_standard=crc.Crc8.MAXIM_DOW, endianness='little', buffer_size=10):
        self.cfg = cfg
        self.crc_standard = crc_standard
        self.endianness = endianness
        
        self.crc_calculator = crc.Calculator(self.crc_standard, optimized=True)
        
        self.serial_port = self.guess_uart_device_()

        self.circular_buffer = []
        self.buffer_size = buffer_size

        self.enemy_color = self.cfg.DEFAULT_ENEMY_TEAM
        self.stm32_color = 'red' if self.enemy_color == 'blue' else 'blue'
        
        self.seq_num = 0
    
    def is_valid(self):
        return self.serial_port is not None
    
    def try_read_one(self):
        # Read from serial port, if any packet is waiting
        if self.serial_port is not None:
            if (self.serial_port.inWaiting() > 0):
                # read the bytes and convert from binary array to ASCII
                byte_array = self.serial_port.read(self.serial_port.inWaiting())

                for c in byte_array:
                    if len(self.circular_buffer) >= self.buffer_size:
                        self.circular_buffer = self.circular_buffer[1:] # pop first element
                    self.circular_buffer.append(c)
    
    def process_one_packet(self, header, yaw_offset, pitch_offset):
        packet = self.create_packet(header, yaw_offset, pitch_offset)
        self.send_packet(packet)
    
    def send_packet(self, packet):
        if self.serial_port is not None:
            self.serial_port.write(packet)
    
    @staticmethod
    def guess_uart_device_():
        # This function is for UNIX-like systems only!
        
        # OSX prefix: "tty.usbmodem"
        # Jetson / Linux prefix: "ttyUSB", "ttyACM"
        UART_PREFIX_LIST = ("tty.usbmodem", "ttyUSB", "ttyACM")

        dev_list = os.listdir("/dev")
        
        serial_port = None # ret val

        for dev_name in dev_list:
            if dev_name.startswith(UART_PREFIX_LIST):
                try:
                    dev_path = os.path.join("/dev", dev_name)
                    serial_port = serial.Serial(
                        port=dev_path,
                        baudrate=115200,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                    )
                except serial.serialutil.SerialException:
                    serial_port = None
                
                if serial_port is not None:
                    return serial_port
        
        print("NO SERIAL DEVICE FOUND! WRITING TO VACCUM!")
        
        return serial_port
        
    def create_packet(self, header, yaw_offset, pitch_offset):
        """
        Packet struct

        Big endian

        HEADER    (2 bytes chars)
        SEQNUM    (4 bytes uint32; wrap around)
        REL_YAW   (4 bytes int32; radians * 1000000/1e+6)
        REL_PITCH (4 bytes int32; radians * 1000000/1e+6)
        CRC8      (1 byte  uint8; CRC checksum MAXIM_DOW of contents BEFORE CRC)
                  (i.e., CRC does not include itself and PACK_END!)
        PACK_END  (2 bytes chars)
        
        Total     (17 bytes)
        """
        assert header in [self.cfg.SEARCH_TARGET, self.cfg.MOVE_YOKE]
        packet = header
        assert isinstance(self.seq_num, int) and self.seq_num >= 0
        if self.seq_num >= 2 ** 32:
            self.seq_num = self.seq_num % (2 ** 32)
        packet += self.seq_num.to_bytes(4, self.endianness)

        discrete_yaw_offset = int(yaw_offset * 1e+6)
        discrete_pitch_offset = int(pitch_offset * 1e+6)

        # TODO: add more sanity check here?
        packet += (discrete_yaw_offset & 0xFFFFFFFF).to_bytes(4, self.endianness)
        packet += (discrete_pitch_offset & 0xFFFFFFFF).to_bytes(4, self.endianness)

        # Compute CRC
        crc8_checksum = self.crc_calculator.checksum(packet)
        assert crc8_checksum >= 0 and crc8_checksum < 256

        packet += crc8_checksum.to_bytes(1, self.endianness)

        # ENDING
        packet += self.cfg.PACK_END

        return packet

    def get_current_stm32_state(self):
        # Decode packet sent from the STM32 controller
        # TODO: if a robot is revived, the serial port might get
        # garbage value in between...

        # TODO: implement a proper CRC-verified packet decoder
        blue_cnt = 0
        red_cnt = 0

        for l in self.circular_buffer:
            if l == ord('R'): red_cnt += 1
            if l == ord('B'): blue_cnt += 1
        
        if blue_cnt > red_cnt:
            self.stm32_color = 'blue'
            self.enemy_color = 'red'
        
        if red_cnt > blue_cnt:
            self.stm32_color = 'red'
            self.enemy_color = 'blue'
        
        ret_dict = {
            'my_color': self.stm32_color,
            'enemy_color': self.enemy_color,
        }

        return ret_dict
