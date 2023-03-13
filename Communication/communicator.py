"""Host the UART communicator. See UARTCommunicator."""
import os
import serial
import crc
import time
import threading

class UARTCommunicator:
    """USB-TTL-UART communicator for Jetson-STM32 communication."""

    def __init__(
            self,
            cfg,
            crc_standard=crc.Crc8.MAXIM_DOW,
            endianness='little',
            buffer_size=10):
        """Initialize the UART communicator.

        Args:
            cfg (python object): shared config
            crc_standard (crc.Crc8): CRC standard
            endianness (str): endianness of the packet. Either 'big' or 'little'
            buffer_size (int): size of the circular buffer
        """
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
        """Return if communicator is valid."""
        return self.serial_port is not None

    def try_read_one(self):
        """Try to read one packet from the serial port and store to internal buffer."""
        # Read from serial port, if any packet is waiting
        if self.serial_port is not None:
            if (self.serial_port.inWaiting() > 0):
                # read the bytes and convert from binary array to ASCII
                byte_array = self.serial_port.read(
                    self.serial_port.inWaiting())

                for c in byte_array:
                    if len(self.circular_buffer) >= self.buffer_size:
                        # pop first element
                        self.circular_buffer = self.circular_buffer[1:]
                    self.circular_buffer.append(c)

    def process_one_packet(self, header, yaw_offset, pitch_offset):
        """Process a batch of numbers into a CRC-checked packet and send it out.
        
        Args:
            header (str): either 'ST' or 'MV'
            yaw_offset (float): yaw offset in radians
            pitch_offset (float): pitch offset in radians
        """
        packet = self.create_packet(header, yaw_offset, pitch_offset)
        self.send_packet(packet)

    def send_packet(self, packet):
        """Send a packet out."""
        if self.serial_port is not None:
            self.serial_port.write(packet)

    @staticmethod
    def guess_uart_device_():
        """Guess the UART device path and open it.
        
        Note: this function is for UNIX-like systems only!

        OSX prefix: "tty.usbmodem"
        Jetson / Linux prefix: "ttyUSB", "ttyACM"

        Returns:
            serial.Serial: the serial port object
        """
        # list of possible prefixes
        UART_PREFIX_LIST = ("tty.usbmodem", "ttyUSB", "ttyACM")

        dev_list = os.listdir("/dev")

        serial_port = None  # ret val

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
        Create CRC-checked packet from user input.

        Args:
            header (str): either 'ST' or 'MV'
            yaw_offset (float): yaw offset in radians
            pitch_offset (float): pitch offset in radians

        Returns:
            bytes: the packet to be sent to serial

        For more details, see doc/comms_protocol.md, but here is a summary of the packet format:

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
        packet += (self.seq_num & 0xFFFFFFFF).to_bytes(4, self.endianness)

        discrete_yaw_offset = int(yaw_offset * 1e+6)
        discrete_pitch_offset = int(pitch_offset * 1e+6)

        # TODO: add more sanity check here?
        packet += (discrete_yaw_offset &
                   0xFFFFFFFF).to_bytes(4, self.endianness)
        packet += (discrete_pitch_offset &
                   0xFFFFFFFF).to_bytes(4, self.endianness)

        # Compute CRC
        crc8_checksum = self.crc_calculator.checksum(packet)
        assert crc8_checksum >= 0 and crc8_checksum < 256

        packet += crc8_checksum.to_bytes(1, self.endianness)

        # ENDING
        packet += self.cfg.PACK_END

        self.seq_num += 1

        return packet

    def get_current_stm32_state(self):
        """Read from buffer from STM32 to Jetson and return the current state.
        
        TODO:
            - Decode packet sent from the STM32 controller
            - If a robot is revived, the serial port might get garbage value in between
            - implement a proper CRC-verified packet decoder

        Returns:
            dict: a dictionary containing the current state of the STM32
        """
        blue_cnt = 0
        red_cnt = 0

        for read_byte in self.circular_buffer:
            if read_byte == ord('R'):
                red_cnt += 1
            if read_byte == ord('B'):
                blue_cnt += 1

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


if __name__ == '__main__':
    # Testing example if run as main
    import sys
    import os
    # setting path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import config
    uart = UARTCommunicator(config)
    for i in range(1000):
        time.sleep(0.005)  # simulate 200Hz
        uart.process_one_packet(config.SEARCH_TARGET, 0.0, 0.0)
