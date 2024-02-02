#!/usr/bin/env python3
#
#Host the UART communicator. See UARTCommunicator.
#
import os
import sys
import logging
import serial
import crc
import time
import threading
import struct
from copy import deepcopy

logger = logging.getLogger(__name__)

# STM32 to Jetson packet size
STJ_MAX_PACKET_SIZE = 21
STJ_MIN_PACKET_SIZE = 10


class MiniPCCommunicationError(Exception):
    pass


class Communicator:
    pass


class UARTCommunicator:
    """USB-TTL-UART communicator for Jetson-STM32 communication."""

    def __init__(
            self,
            cfg,
            crc_standard=crc.Crc8.MAXIM_DOW,
            endianness='little',
            warn=True,
            serial_dev_path=None,  # None -> guess port, False -> port=None
            serial_dev=None,       # None -> ^^^^^^^^^^
            in_use=None,
            buffer_size=STJ_MAX_PACKET_SIZE * 100):
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

        self.warn = warn

        if in_use is None:
            if self.warn:
                logger.warning('Did not receive a list of ports in use, assuming none.')
            in_use = []

        if serial_dev is None:
            if serial_dev_path is None:
                self.use_uart_device(self.guess_uart_device_(in_use=in_use), in_use=in_use)
            elif not serial_dev_path:
                self.use_uart_device_path(None, in_use=in_use)
            else:
                if serial_dev_path == '/dev/tty' and self.warn:
                    logger.warn('Using special device file `/dev/tty\': is this really expected?')
                self.use_uart_device_path(serial_dev_path, in_use=in_use)
        else:
            self.use_uart_device_path(serial_dev, in_use=in_use)


        self.circular_buffer = []
        self.buffer_size = buffer_size

        self.stm32_state_dict = {
            'my_color': 'red' if self.cfg.DEFAULT_ENEMY_TEAM == 'blue' else 'blue',
            'enemy_color': self.cfg.DEFAULT_ENEMY_TEAM.lower(),
            'rel_yaw': 0,
            'rel_pitch': 0,
            'debug_int': 0,
            'mode': 'ST',
            'vx': 0,
            'vy': 0,
            'vw': 0,
        }

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
        """Try to copy from serial port to a circular buffer.

        Returns:
            bool: True if there are data waiting in the serial port
        """
        # Read from serial port, if any packet is waiting
        if self.serial_port is not None:
            if self.serial_port.inWaiting() > 0:
                # read the bytes and convert from binary array to ASCII
                byte_array = self.serial_port.read(self.serial_port.inWaiting())

                for c in byte_array:
                    if len(self.circular_buffer) >= self.buffer_size:
                        # pop first element
                        self.circular_buffer = self.circular_buffer[1:]
                    self.circular_buffer.append(c)
                return True
            else:
                return False

    def create_and_send_packet(self, cmd_id, data):
        """Process a batch of numbers into a CRC-checked packet and send it out.

        Args:
            cmd_id (int): see config.py
            data (dict): all key and values are defined in the docs/comm_protocol.md
            Here's a list of cmd_id's and their data for quick reference
            For a more detailed description, see docs/comm_protocols.md
            cmd_id == GIMBAL_CMD_ID:
            data = {'rel_yaw': float, 'rel_pitch': float, 'mode': 'ST' or 'MY',
                    'debug_int': uint8_t}

            cmd_id == COLOR_CMD_ID:
              data = {'my_color': 'red' or 'blue', 'enemy_color': same as my_color}

            cmd_id == CHASSIS_CMD_ID:
              data = {'vx': float, 'vy': float, 'vw': float}
        """
        packet = self.create_packet(cmd_id, data)
        self.send_packet(packet)

    def send_packet(self, packet):
        """Send a packet out."""
        if self.serial_port is not None:
            self.serial_port.write(packet)

    def use_uart_device_path(self, dev_path, in_use):
        if dev_path in in_use:
            logger.warning('{dev_path} already in use: is this really expected?')
        dev = UARTCommunicator.try_uart_device(dev_path, in_use)
        if dev is None:
            if self.warn:
                logger.warning("NO SERIAL DEVICE FOUND! WRITING TO VACCUM!")
        else:
            in_use += [dev.port]
        logger.info(f'I ({self.__class__=}) am using {dev}.')
        self.serial_port = dev

    def use_uart_device(self, dev, in_use):
        try:
            if dev.port in in_use:
                logger.warning('{dev.port} already in use: is this really expected?')
            else:
                in_use += [dev.port]
        except AttributeError:  # dev is None
            logger.warning("NO SERIAL DEVICE FOUND! WRITING TO VACCUM!")
        finally:
            logger.info(f'I ({self.__class__=}) am using {dev}.')
            self.serial_port = dev

    @staticmethod
    def list_uart_device_paths():
        """Guess the UART device paths and return them.

        Note: this function is for UNIX-like systems only!

        OSX prefix: "tty.usbmodem"
        Jetson / Linux prefix: "ttyUSB", "ttyACM"

        Returns:
            [Maybe dev_path] : a list of possible device paths
        """
        # list of possible prefixes
        UART_PREFIX_LIST = ("tty.usbmodem", "ttyUSB", "ttyACM")
        dev_list = os.listdir("/dev")

        dev_paths = []  # ret val
        for dev_name in dev_list:
            if dev_name.startswith(UART_PREFIX_LIST):
                dev_paths += [os.path.join('/dev', dev_name)]
        return dev_paths or [None]

    @staticmethod
    # path -> [path] -> Maybe serial.Serial
    def try_uart_device(dev_path, in_use):
        if dev_path in in_use:
            logger.error(f'Path {dev_path} already in use, returning None.')
            return None
        # Fails with serial.serialutil.SerialException
        serial_port = serial.Serial(
            port=dev_path,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        if serial_port.port is not None:
            logger.info(f'Successfully opened serial on path: {dev_path}')
            return serial_port
        else:
            logger.info(f'Failed to open serial on path: {dev_path}, '
                         'returning None object instead.')
            return None

    @staticmethod
    # [path] -> Maybe serial.Serial
    def guess_uart_device_(in_use):
        """Guess the UART device path and open it.

        Note: this function is for UNIX-like systems only!

        OSX prefix: "tty.usbmodem"
        Jetson / Linux prefix: "ttyUSB", "ttyACM"

        Returns:
            serial.Serial: the serial port object
        """
        logger.info('I will now try to guess a uart device.')
        # list of possible prefixes
        dev_paths = UARTCommunicator.list_uart_device_paths()

        serial_port = None  # ret val
        for dev_path in dev_paths:
            if dev_path in in_use:
                logger.info(f'Guessed {dev_path} but it is already in use, skipping.')
                continue
            if dev_path is not None:
                try:
                    serial_port = UARTCommunicator.try_uart_device(dev_path, in_use=in_use)
                    if serial_port is not None:
                        return serial_port
                except serial.serialutil.SerialException:
                    print('Could not open serial port, skipping...')
        return serial_port

    def packet_search(self):
        """Parse internal circular buffer.

        Returns: True if a valid packet is found
        """
        start_idx = 0
        packet_found = False
        while start_idx <= len(self.circular_buffer) - STJ_MAX_PACKET_SIZE:
            header_letters = (
                self.circular_buffer[start_idx], self.circular_buffer[start_idx + 1])
            if header_letters == (self.cfg.PACK_START[0], self.cfg.PACK_START[1]):
                # Try to parse
                possible_packet = self.circular_buffer[start_idx:start_idx + STJ_MAX_PACKET_SIZE]
                ret_dict = self.try_parse_one(possible_packet)
                if ret_dict is not None:
                    # Successfully parsed one
                    self.parsed_packet_cnt += 1
                    with self.state_dict_lock:
                        self.update_current_state(ret_dict)
                    # Remove parsed bytes from the circular buffer
                    self.circular_buffer = self.circular_buffer[start_idx + (
                        self.cfg.CMD_TO_LEN[ret_dict['cmd_id']] + self.cfg.HT_LEN):]
                    packet_found = True
                else:
                    self.circular_buffer = self.circular_buffer[start_idx + STJ_MIN_PACKET_SIZE:]
                start_idx = 0
            else:
                start_idx += 1
        return packet_found

    def update_current_state(self, ret_dict):
        """
        Update stm32 state dict.

        Helper function.
        """
        # Dont do self.stm32_state_dict['data'] = ret_dict['data'] because different threads
        # may need different information from the stm32
        if ret_dict['cmd_id'] == self.cfg.GIMBAL_CMD_ID:
            self.stm32_state_dict['rel_yaw'] = ret_dict['data']['rel_yaw']
            self.stm32_state_dict['rel_pitch'] = ret_dict['data']['rel_pitch']
            self.stm32_state_dict['debug_int'] = ret_dict['data']['debug_int']
            self.stm32_state_dict['mode'] = ret_dict['data']['mode']
        elif ret_dict['cmd_id'] == self.cfg.COLOR_CMD_ID:
            self.stm32_state_dict['my_color'] = ret_dict['data']['my_color']
            self.stm32_state_dict['enemy_color'] = ret_dict['data']['enemy_color']
        elif ret_dict['cmd_id'] == self.cfg.CHASSIS_CMD_ID:
            self.stm32_state_dict['vx'] = ret_dict['data']['vx']
            self.stm32_state_dict['vy'] = ret_dict['data']['vy']
            self.stm32_state_dict['vw'] = ret_dict['data']['vw']
        elif ret_dict['cmd_id'] == self.cfg.SELFCHECK_CMD_ID:
            self.stm32_state_dict['rel_yaw'] = ret_dict['data']['rel_yaw']
            self.stm32_state_dict['rel_pitch'] = ret_dict['data']['rel_pitch']
            self.stm32_state_dict['debug_int'] = ret_dict['data']['debug_int']
            self.stm32_state_dict['mode'] = ret_dict['data']['mode']

    def try_parse_one(self, possible_packet):
        """
        Parse a possible packet.

        For details on the struct of the packet, refer to docs/comm_protocol.md

        Args:
            possible_packet (list): a list of bytes

        Returns:
            dict: a dictionary of parsed data; None if parsing failed
        """
        assert len(possible_packet) >= STJ_MIN_PACKET_SIZE
        assert possible_packet[0] == self.cfg.PACK_START[0]
        assert possible_packet[1] == self.cfg.PACK_START[1]

        # Check CMD ID valid
        cmd_id = int(possible_packet[self.cfg.CMD_ID_OFFSET])
        try:
            packet_len = self.cfg.CMD_TO_LEN[cmd_id] + self.cfg.HT_LEN
        except BaseException:
            print("Incorrect CMD_ID " + str(cmd_id))
            return None

        # Check packet end
        if possible_packet[packet_len -
                           2] != self.cfg.PACK_END[0] or possible_packet[packet_len -
                                                                         1] != self.cfg.PACK_END[1]:
            return None

        # Compute checksum
        crc_checksum = self.crc_calculator.checksum(bytes(possible_packet[:packet_len - 3]))
        if crc_checksum != possible_packet[packet_len - 3]:
            print("Packet received but crc checksum is wrong")
            return None

        # Parse data into a dictionary
        data = self.parse_data(possible_packet, cmd_id)

        return {
            'cmd_id': cmd_id,
            'data': data
        }

    def parse_data(self, possible_packet, cmd_id):
        """
        Parse the data section of a possible packet.

        Helper function for details on the struct of the packet, refer to docs/comm_protocol.md

        Args:
            data (map): keys are value types are defined in docs/comm_protocol
        Returns:
            dict: a dictionary of parsed data; None if parsing failed
        """
        data = None
        # Parse Gimbal data, CMD_ID = 0x00
        if cmd_id == self.cfg.GIMBAL_CMD_ID:
            # "<f" means little endian float
            rel_yaw = struct.unpack('<f', bytes(
                possible_packet[self.cfg.DATA_OFFSET + 0: self.cfg.DATA_OFFSET + 4]))[0]
            rel_pitch = struct.unpack('<f', bytes(
                possible_packet[self.cfg.DATA_OFFSET + 4: self.cfg.DATA_OFFSET + 8]))[0]
            mode_int = int(possible_packet[self.cfg.DATA_OFFSET + 8])
            mode = self.cfg.GIMBAL_MODE[mode_int]
            debug_int = int(possible_packet[self.cfg.DATA_OFFSET + 9])
            data = {
                'rel_yaw': rel_yaw,
                'rel_pitch': rel_pitch,
                'mode': mode,
                'debug_int': debug_int}
        # Parse Chassis data, CMD_ID = 0x02
        elif cmd_id == self.cfg.CHASSIS_CMD_ID:
            vx = struct.unpack('<f', bytes(
                possible_packet[self.cfg.DATA_OFFSET + 0: self.cfg.DATA_OFFSET + 4]))[0]
            vy = struct.unpack('<f', bytes(
                possible_packet[self.cfg.DATA_OFFSET + 4: self.cfg.DATA_OFFSET + 8]))[0]
            vw = struct.unpack('<f', bytes(
                possible_packet[self.cfg.DATA_OFFSET + 8: self.cfg.DATA_OFFSET + 12]))[0]
            data = {'vx': vx, 'vy': vy, 'vw': vw}
        # Parse color data, CMD_ID = 0x01
        elif cmd_id == self.cfg.COLOR_CMD_ID:
            # 0 for RED; 1 for BLUE
            my_color_int = int(possible_packet[self.cfg.DATA_OFFSET])
            if my_color_int == 0:
                my_color = 'red'
                enemy_color = 'blue'
            else:
                my_color = 'blue'
                enemy_color = 'red'
            data = {'my_color': my_color, 'enemy_color': enemy_color}
        # Parse Selfcheck data, CMD_ID = 0x03
        if cmd_id == self.cfg.SELFCHECK_CMD_ID:
            # "<f" means little endian float
            rel_yaw = struct.unpack('<f', bytes(
                possible_packet[self.cfg.DATA_OFFSET + 0: self.cfg.DATA_OFFSET + 4]))[0]
            rel_pitch = struct.unpack('<f', bytes(
                possible_packet[self.cfg.DATA_OFFSET + 4: self.cfg.DATA_OFFSET + 8]))[0]
            mode_int = int(possible_packet[self.cfg.DATA_OFFSET + 8])
            mode = self.cfg.SELFCHECK_MODE[mode_int]
            debug_int = int(possible_packet[self.cfg.DATA_OFFSET + 9])
            data = {
                'rel_yaw': rel_yaw,
                'rel_pitch': rel_pitch,
                'mode': mode,
                'debug_int': debug_int}
        return data

    def create_packet(self, cmd_id, data):
        """
        Create CRC-checked packet from user input.

        Args:
            cmd_id (int): see config.py
            data (dict): all key and values are defined in the docs/comm_protocol.md
        Returns:
            bytes: the packet to be sent to serial

        For more details, see doc/comms_protocol.md, but here is a summary of the packet format:
        Little endian

        HEADER    (2 bytes chars)
        SEQNUM    (2 bytes uint16; wrap around)
        DATA_LEN  (2 bytes uint16)
        CMD_ID    (1 byte  uint8)
        DATA      (see docs/comms_protocol.md for details)
        CRC8      (1 byte  uint8; CRC checksum MAXIM_DOW of contents BEFORE CRC)
                  (i.e., CRC does not include itself and PACK_END!)
        PACK_END  (2 bytes chars)
        """
        # Prepare header
        packet = self.cfg.PACK_START
        assert isinstance(self.seq_num, int) and self.seq_num >= 0
        if self.seq_num >= 2 ** 16:
            self.seq_num = self.seq_num % (2 ** 16)
        packet += (self.seq_num & 0xFFFF).to_bytes(2, self.endianness)
        packet += self.cfg.CMD_TO_LEN[cmd_id].to_bytes(1, self.endianness)
        packet += cmd_id.to_bytes(1, self.endianness)

        # Prepare data
        packet += self.create_packet_data(cmd_id, data)

        # Compute CRC
        crc8_checksum = self.crc_calculator.checksum(packet)
        assert crc8_checksum >= 0 and crc8_checksum < 256

        packet += crc8_checksum.to_bytes(1, self.endianness)

        # ENDING
        packet += self.cfg.PACK_END

        self.seq_num += 1
        return packet

    def create_packet_data(self, cmd_id, data):
        """
        Create the data section for a packet.

        Helper function.
        Args:
            cmd_id (int): see config.py
            data (dict): all key and values are defined in the docs/comm_protocol.md
        Returns:
            bytes: the data section of the packet to be sent to serial
        """
        # empty binary string
        packet = b''
        # Parse Gimbal data, CMD_ID = 0x00
        if cmd_id == self.cfg.GIMBAL_CMD_ID:
            # "<f" means little endian float
            packet += struct.pack("<f", data['rel_yaw'])
            packet += struct.pack("<f", data['rel_pitch'])
            # 0 for 'ST' 1 for 'MY',
            packet += self.cfg.GIMBAL_MODE.index(data['mode']).to_bytes(1, self.endianness)
            packet += data['debug_int'].to_bytes(1, self.endianness)
        # Parse Chassis data, CMD_ID = 0x02
        elif cmd_id == self.cfg.CHASSIS_CMD_ID:
            # "<f" means little endian float
            packet += struct.pack("<f", data['vx'])
            packet += struct.pack("<f", data['vy'])
            packet += struct.pack("<f", data['vw'])
        # Parse color data, CMD_ID = 0x01
        elif cmd_id == self.cfg.COLOR_CMD_ID:
            # 0 for RED; 1 for BLUE
            if data['my_color'] == 'red':
                my_color_int = 0
            elif data['my_color'] == 'blue':
                my_color_int = 1
            packet += my_color_int.to_bytes(1, self.endianness)
        # Parse Self Test data, CMD_ID = 0x00
        # Temp. same as gimbal
        if cmd_id == self.cfg.SELFCHECK_CMD_ID:
            # "<f" means little endian float
            packet += struct.pack("<f", data['rel_yaw'])
            packet += struct.pack("<f", data['rel_pitch'])
            # 0 for 'ST' 1 for 'MY',
            packet += self.cfg.SELFCHECK_MODE.index(data['mode']).to_bytes(1, self.endianness)
            packet += data['debug_int'].to_bytes(1, self.endianness)
        # Data length = Total length - 9
        assert len(packet) == self.cfg.CMD_TO_LEN[cmd_id]
        return packet

    def get_current_stm32_state(self):
        """Read from buffer from STM32 to Jetson and return the current state."""
        with self.state_dict_lock:
            ret_dict = deepcopy(self.stm32_state_dict)
        return ret_dict


class USBCommunicator:
    def __init__(self):
        pass


# XXX: Move tests out, leave simpler unit tests? i.e. only pingpong
# Latency test by Richard, flash example/minipc/LatencyTest.cc
# Modified by Austin.
# Tests Minipc <-> Type C board circuit time
def test_board_latency(uart, rounds=15, timeout=1, hz=200,
                 verbose=True):
    #print(logger.getLevelName(logger.getEffectiveLevel()))
    print('Communicator beginning minipc <-> board latency test: '
          f'{rounds} rounds at {hz} Hertz')
    cmd_id = uart.cfg.SELFCHECK_CMD_ID
    def send_packets(rounds, hz):
        send_time = [0] * rounds
        packet_status = [False] * rounds
        for i in range(rounds):
            logger.debug(f'Sending packet #{i} to stm32...')
            data = {'rel_yaw': i, 'rel_pitch': i,
                    'mode': 'EC', 'debug_int' : i}

            send_time[i] = time.time()
            uart.create_and_send_packet(cmd_id, data)
            packet_status[i] = True

            time.sleep(1/hz)

        return (send_time, packet_status)

    def receive_packets(rounds, timeout, ret):  # Async
        received = 0
        receive_time = [0] * rounds
        packet_status = [False] * rounds
        # Receive loop
        current_time = time.time()
        while time.time() - current_time < timeout and received != rounds:
            uart.try_read_one()
            if uart.packet_search():
                received_data = uart.get_current_stm32_state()
                i = int(received_data['rel_yaw'])
                # rel_yaw and rel_pitch act as (the same) index,
                receive_time[i] = time.time()
                logger.debug(f'Received packet #{i} from stm32...')
                received += 1
            time.sleep(0.001) # Use same frequency as listen_.
        for i, t in enumerate(receive_time):
            if t != 0:
                packet_status[i] = True

        ret[0:1] = [receive_time, packet_status]
        return ret[0:1] # If not run as Thread.

    # Start the receive thread first
    rt_return = []
    receive_thread = threading.Thread(target=receive_packets,
                                      args=(rounds, timeout, rt_return))
    receive_thread.start()
    # Send packets second
    send_time, send_packet_status = send_packets(rounds, hz)
    receive_thread.join()
    receive_time, receive_packet_status = rt_return
    # Flatten data
    not_all_received = not all(receive_packet_status)
    # 0 if packet not received
    latencies = [(tf or ti) - ti
                 for ti, tf in zip(send_time, receive_time)]
    statuses = [*zip(send_packet_status, receive_packet_status)]

    loss = latencies.count(0.0)
    average_latency = sum(latencies)/(len(latencies)-loss or 1) # Prevent 0/0

    for i in range(rounds):
        is_sent = statuses[i][0]
        is_received = statuses[i][1]
        logger.debug('Status of packet %d: send: %s, receive: %s' %
               (i, ('UNSENT!', 'sent')[is_sent],
                   ('NOT RECEIVED!', 'received')[is_received]))
        logger.debug(f'Latency of packet #{i}: {latencies[i]}')

    print('\nAttempted to send', rounds, 'packets.',
           send_packet_status.count(True), 'Packets transmitted,',
           rounds-loss, 'packets received.')
    print(f'Packets lost: {loss}/{loss/rounds*100}%. '
          f'Average latency: {average_latency}')
    if not_all_received:
        logger.warning('Latency test: not all packets were received.')

    return {'average' : average_latency,
            'loss' : (loss, loss/rounds),
            'detailed' : [*zip(statuses, latencies)]}

# RX/TX test by YHY modified by Richard, flash example/minipc/PingPongTest.cc
# this ping pong test first trys to send a packet
# and then attmepts to read the response from stm32 for 10 seconds
# then send a second packet
# after that entering ping pong mode:
#   receive packet from stm32, rel_pitch += 1 then immediately send back
# each "ping pong" has a ID for differentiating during pingping-ing
# TODO: This test shows the issue that a response can only be received after the data
# in circular_buffer is at least the maximum size of a packet (STJ_MAX_PACKET_SIZE).
# So if sending some small packets,
# they will stay in the circular_buffer waiting to be parsed,
# until new packets are received.
# For example, if STJ_MAX_PACKET_SIZE is 21 and GIMBAL data size is 19,
# then only after receiving 2 packets (2 * 19 > 21)
# then the first packet will be parsed.
# If a data type is 10 bytes long then sending a third packet is necessary
# before pingpong
def test_board_pingpong():
    i = 0
    cmd_id = uart.cfg.GIMBAL_CMD_ID
    packet_count = 0

    # sending packet the first time
    data = {'rel_yaw': packet_count, 'rel_pitch': 0.0, 'mode': 'ST', 'debug_int': 42}
    print("Sending the first time: ID = " +
          str(data['rel_yaw']) + " count = " + str(data['rel_pitch']))
    uart.create_and_send_packet(cmd_id, data)

    # attempt to receive packet
    print('attemp to receive and parse response...')
    for i in range(20):
        time.sleep(0.1)
        if uart.try_read_one():
            print("data waiting at serial port")
            print("circular buffer:" + str(uart.circular_buffer))
        if uart.packet_search():
            received_data = uart.get_current_stm32_state()
            print("from stm32: rel_yaw = " +
                  str(received_data['rel_yaw']) +
                  " rel_pitch = " +
                  str(received_data['rel_pitch']))

    # sending packet the second time
    packet_count += 1
    data = {'rel_yaw': packet_count, 'rel_pitch': 0.0, 'mode': 'ST', 'debug_int': 42}
    print("Sending the second time: ID = " +
          str(data['rel_yaw']) + " count = " + str(data['rel_pitch']))
    uart.create_and_send_packet(cmd_id, data)

    print('starting ping pong')
    # start ping pong
    while True:
        uart.try_read_one()
        # update stm32 status from packet from stm32
        if uart.packet_search():
            received_data = uart.get_current_stm32_state()
            print("from stm32: rel_yaw = " +
                  str(received_data['rel_yaw']) +
                  " rel_pitch = " +
                  str(received_data['rel_pitch']))
            received_data['rel_pitch'] = received_data['rel_pitch'] + 1
            uart.create_and_send_packet(cmd_id, received_data)

        time.sleep(0.05)

# rate test by Roger modified by Richard, flash example/minipc/StressTestTypeC.cc
# Modified by Austin.
# TODO: currently this test will never receive full 1000 packets
#    but only 998 packets because the last two packets
#    remain in circular buffer and not parsed because
#    its size is not reaching STJ_MAX_PACKET_SIZE
# NOTE: please reflash or restart program on stm32 every time you want to run this test
# TODO: Notify the board and use COLOR packets instead?
def test_board_crc(uart, rounds=15, timeout=1, hz=200,
             verbose=True):
    print('Communicator beginning minipc <-> board crc stress test: '
          f'{rounds} rounds at {hz} Hertz')
    cmd_id = uart.cfg.SELFCHECK_CMD_ID
    def send_packets(rounds, hz):
        packet_status = [False] * rounds
        for i in range(rounds):
            logger.debug(f'Sending packet #{i} to stm32...')
            data = {'rel_yaw': i, 'rel_pitch': i,
                    'mode': 'EC', 'debug_int' : 0}

            uart.create_and_send_packet(cmd_id, data)
            packet_status[i] = True

            time.sleep(1/hz)

        return packet_status

    def receive_packets(rounds, timeout, ret): # Async
        received = 0
        packet_status = [False] * rounds
        # Receive loop
        current_time = time.time()
        while time.time() - current_time < timeout and received != rounds:
            uart.try_read_one()
            if uart.packet_search():
                received_data = uart.get_current_stm32_state()
                i = int(received_data['rel_yaw'])
                # rel_yaw and rel_pitch act as (the same) index,
                # debug_int now contains cmd_id
                packet_status[i] = True
                logger.debug(f'Received packet #{i} from stm32...')
                received += 1
            time.sleep(0.001) # Use same frequency as listen_.

        ret[0] = packet_status
        return ret[0] # If not run as Thread.

    # Send packets first
    send_packet_status = send_packets(rounds, hz)
    print(f'Packet sending test complete: sent {rounds} packets.')
    print('You should see the light change from blue to green on type C board.')
    print('When the led turns red the stm32 is sending data.')
    print('Starting packet receiving test.')
    # Start the receive thread second
    rt_return = [None]
    receive_thread = threading.Thread(target=receive_packets,
                                      args=(rounds, timeout, rt_return))
    receive_thread.start()
    receive_thread.join()
    receive_packet_status = rt_return[0]
    # Flatten data
    not_all_received = not all(receive_packet_status)
    statuses = [*zip(send_packet_status, receive_packet_status)]

    loss = receive_packet_status.count(False)

    print(f'\nAttempted to send {rounds} packets: '
          f'{send_packet_status.count(True)} packets transmitted, '
          f'{rounds-loss} packets received.')
    print(f'Packets lost: {loss}/{loss/rounds*100}%.')

    if not_all_received:
        logger.warning('Crc test: not all packets were received.')

    return {'loss' : (loss, loss/rounds),
            'detailed' : statuses}

def test_board_typea():
    # vanilla send test, flash typeA.cc
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

if __name__ == '__main__':
    # Unit testing
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from enum import Enum
    import config

    uart = UARTCommunicator(config)

    class Test(Enum):
        """Class used to choose test for communicator."""
        LATENCY = 1
        PINGPONG = 2
        CRC = 3
        TYPE_A = 4

    # Remove first arg if called with python.
    if 'python' in sys.argv[0]:
        sys.argv.pop(0)

    testing = Test.PINGPONG
    if len(sys.argv) > 1:
        testing = (Test.LATENCY, Test.PINGPONG,
                   Test.CRC, Test.TYPE_A)[int(sys.argv[1]) - 1]
        print(f'\nUsing test type: {testing}')
    else:
        print(f'\nUsing default test type: {testing}')
    print("Change test type: ./communicator.py [1|2|3|4]")
    print("1: LATENCY, 2: PINGPONG, 3: CRC, 4: TYPE_A\n")

    match testing:
        case Test.LATENCY:
            test_board_latency()
        case Test.PINGPONG:
            test_board_pingpong()
        case Test.CRC:
            test_board_crc()
        case Test.TYPE_A:
            test_board_typea()
        case _:
            print("Invalid selection")

