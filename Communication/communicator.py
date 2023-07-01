"""Host the UART communicator. See UARTCommunicator."""
import os
import serial
import crc
import time
import threading
import platform
from copy import deepcopy

# STM32 to Jetson packet size
STJ_PACKET_SIZE = 18
INT_FP_SCALE = 1e+6


class UARTCommunicator:
    """USB-TTL-UART communicator for Jetson-STM32 communication."""

    def __init__(
            self,
            cfg,
            crc_standard=crc.Crc8.MAXIM_DOW,
            endianness='little',
            buffer_size=STJ_PACKET_SIZE * 100):
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

        self.stm32_state_dict = {
            'my_color': 'red' if self.cfg.DEFAULT_ENEMY_TEAM == 'blue' else 'blue',
            'enemy_color': self.cfg.DEFAULT_ENEMY_TEAM.lower(),
            'cur_yaw': 0,
            'cur_pitch': 0,
            'debug_int': 0}

        self.parsed_packet_cnt = 0
        self.seq_num = 0

        self.state_dict_lock = threading.Lock()

    def start_listening(self):
        """Start a thread to listen to the serial port."""
        self.listen_thread = threading.Thread(target=self.listen_)
        self.listen_thread.start()

    def listen_(self, interval=0.001):
        """
        Listen to the serial port.

        This function updates circular_buffer and stm32_state_dict.

        TODO: test this function on real jetson / STM32!

        Args:
            interval (float): interval between two reads
        """
        while True:
            self.try_read_one()
            self.packet_search()
            time.sleep(interval)

    def is_valid(self):
        """Return if communicator is valid."""
        return self.serial_port is not None

    def try_read_one(self):
        """
        Try to read one packet from the serial port and store to internal buffer.

        Returns:
            bool: True if a packet is read; False otherwise
        """
        # Read from serial port, if any packet is waiting
        if self.serial_port is not None:
            if (self.serial_port.inWaiting() > 0):
                # read the bytes and convert from binary array to ASCII
                try:
                    byte_array = self.serial_port.read(self.serial_port.inWaiting())
                except serial.serialutil.SerialException:
                    print("No data returned?")
                    return False

                for c in byte_array:
                    if len(self.circular_buffer) >= self.buffer_size:
                        # pop first element
                        self.circular_buffer = self.circular_buffer[1:]
                    self.circular_buffer.append(c)
                return True
            else:
                return False

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

        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            dev_list = os.listdir("/dev")
        elif platform.system() == 'Windows':
            print("Windows is not yet supported for serial driver!")
            print("Writing to vaccum!")
            return None
        else:
            print("Unknown system: {}".format(platform.system()))
            print("Writing to vaccum!")
            return None

        serial_port = None  # ret val

        for dev_name in dev_list:
            if dev_name.startswith(UART_PREFIX_LIST):
                try:
                    print("Trying to open serial port: {}".format(dev_name))
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

    def packet_search(self):
        """Parse internal circular buffer."""
        start_idx = 0
        while start_idx <= len(self.circular_buffer) - STJ_PACKET_SIZE:
            header_letters = (
                self.circular_buffer[start_idx], self.circular_buffer[start_idx + 1])
            if header_letters == (ord('H'), ord('D')):
                # Try to parse
                possible_packet = self.circular_buffer[start_idx:start_idx + STJ_PACKET_SIZE]
                ret_dict = self.try_parse_one(possible_packet)
                if ret_dict is not None:
                    # Successfully parsed one
                    self.parsed_packet_cnt += 1
                    self.state_dict_lock.acquire()
                    self.stm32_state_dict = ret_dict
                    self.state_dict_lock.release()
                # Remove parsed bytes from the circular buffer
                self.circular_buffer = self.circular_buffer[start_idx + STJ_PACKET_SIZE:]
                start_idx = 0
            else:
                start_idx += 1

    def try_parse_one(self, possible_packet):
        """
        Parse a possible packet.

        For details on the struct of the packet, refer to docs/comm_protocol.md

        Args:
            possible_packet (list): a list of bytes

        Returns:
            dict: a dictionary of parsed data; None if parsing failed
        """
        assert len(possible_packet) == STJ_PACKET_SIZE
        assert possible_packet[0] == ord('H')
        assert possible_packet[1] == ord('D')

        # Check packet end
        if possible_packet[-2] != ord('E') or possible_packet[-1] != ord('D'):
            return None

        # Compute checksum
        crc_checksum = self.crc_calculator.checksum(bytes(possible_packet[:-3]))
        if crc_checksum != possible_packet[-3]:
            return None

        # Valid packet

        # 0 for RED; 1 for BLUE
        my_color_int = int(possible_packet[2])

        cur_yaw = int.from_bytes(possible_packet[3:7], "little", signed=True) / INT_FP_SCALE
        cur_pitch = int.from_bytes(possible_packet[7:11], "little", signed=True) / INT_FP_SCALE
        debug_int = int.from_bytes(possible_packet[11:15], "little", signed=True)

        if my_color_int == 0:
            my_color = 'red'
            enemy_color = 'blue'
        else:
            my_color = 'blue'
            enemy_color = 'red'

        return {
            'my_color': my_color,
            'enemy_color': enemy_color,
            'cur_yaw': cur_yaw,
            'cur_pitch': cur_pitch,
            'debug_int': debug_int,
        }

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

        discrete_yaw_offset = int(yaw_offset * INT_FP_SCALE)
        discrete_pitch_offset = int(pitch_offset * INT_FP_SCALE)

        # TODO: add more sanity check here?
        packet += (discrete_yaw_offset & 0xFFFFFFFF).to_bytes(4, self.endianness)
        packet += (discrete_pitch_offset & 0xFFFFFFFF).to_bytes(4, self.endianness)

        # Compute CRC
        crc8_checksum = self.crc_calculator.checksum(packet)
        assert crc8_checksum >= 0 and crc8_checksum < 256

        packet += crc8_checksum.to_bytes(1, self.endianness)

        # ENDING
        packet += self.cfg.PACK_END

        self.seq_num += 1

        return packet

    def get_current_stm32_state(self):
        """Read from buffer from STM32 to Jetson and return the current state."""
        self.state_dict_lock.acquire()
        ret_dict = deepcopy(self.stm32_state_dict)
        self.state_dict_lock.release()
        return ret_dict


if __name__ == '__main__':
    TESTING_CRC = False
    # Testing example if run as main
    import sys
    import os
    # setting path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import config
    uart = UARTCommunicator(config)

    if TESTING_CRC:
        print("Starting packet sending test.")
        for i in range(1000):
            time.sleep(0.005)  # simulate 200Hz
            uart.process_one_packet(config.SEARCH_TARGET, 0.0, 0.0)

        print("Packet sending test complete.")
        print("You should see the light change from bllue to green on type C board.")
        print("Starting packet receiving test.")

        while True:
            if uart.parsed_packet_cnt == 1000:
                print("Receiver successfully parsed exactly 1000 packets.")
                break
            if uart.parsed_packet_cnt > 1000:
                print("Repeatedly parsed one packet?")
                break
            uart.try_read_one()
            uart.packet_search()
            time.sleep(0.001)

        print(uart.get_current_stm32_state())
        print("Packet receiving test complete.")
    else:
        cur_packet_cnt = uart.parsed_packet_cnt
        cur_time = time.time()
        prv_parsed_packet_cnt = 0
        while True:
            uart.try_read_one()
            uart.packet_search()
            if uart.parsed_packet_cnt > cur_packet_cnt:
                cur_packet_cnt = uart.parsed_packet_cnt
                # print(uart.get_current_stm32_state())
            time.sleep(0.001)
            if time.time() > cur_time + 1:
                print("Parsed {} packets in 1 second.".format(
                    cur_packet_cnt - prv_parsed_packet_cnt))
                prv_parsed_packet_cnt = cur_packet_cnt
                cur_time = time.time()
        # while True:
        #     time.sleep(0.005)
        #     uart.process_one_packet(config.SEARCH_TARGET, 0.01, 0.0)
