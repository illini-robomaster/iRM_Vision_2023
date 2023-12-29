"""Host the UART communicator. See UARTCommunicator."""
import os
import serial
import crc
import time
import threading
import struct
from copy import deepcopy

from enum import Enum

    

# STM32 to Jetson packet size
STJ_MAX_PACKET_SIZE = 21
STJ_MIN_PACKET_SIZE = 10


class UARTCommunicator:
    """USB-TTL-UART communicator for Jetson-STM32 communication."""

    def __init__(
            self,
            cfg,
            crc_standard=crc.Crc8.MAXIM_DOW,
            endianness='little',
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

        self.serial_port = self.guess_uart_device_()

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
                    self.state_dict_lock.acquire()
                    self.update_current_state(ret_dict)
                    self.state_dict_lock.release()
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

    def get_circular_buffer(self):
        return self.circular_buffer[:]

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
        if cmd_id in self.cfg.CMD_TO_LEN.keys():
            packet_len = self.cfg.CMD_TO_LEN[cmd_id] + self.cfg.HT_LEN  #"KeyError: 84"?
        else:
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
        # Data length = Total length - 9
        assert len(packet) == self.cfg.CMD_TO_LEN[cmd_id]
        return packet

    def get_current_stm32_state(self):
        """Read from buffer from STM32 to Jetson and return the current state."""
        self.state_dict_lock.acquire()
        ret_dict = deepcopy(self.stm32_state_dict)
        self.state_dict_lock.release()
        return ret_dict


class Test(Enum):
    LATENCY=1
    PINGPONG=2
    CRC=3
    TYPE_A=4
    MODIFIED_PINGPONG=5
    COLOR=6
    CHASSIS=7


if __name__ == '__main__':
    testing = Test.CRC
    # Testing example if run as main
    import sys
    import os
    # setting path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import config
    uart = UARTCommunicator(config)

    match testing:
        #Test for CHASSIS packet by Richard, flash exampels/minipc/ChassisTest.cc
        case Test.CHASSIS:
            cmd_id = uart.cfg.CHASSIS_CMD_ID

            data = {'vx': 10.0, 'vy': 11.0, 'vw' : 12.0}
            print("sending color data:" + str(data))
            uart.create_and_send_packet(cmd_id, data)
            send_time = time.time()

            received = False
            #trying to read response for 1 s
            while time.time()-send_time < 1:
                if uart.try_read_one():
                    print(uart.get_circular_buffer())
                if uart.packet_search(): 
                    received_data = uart.get_current_stm32_state()
                    latency = time.time()-send_time
                    print("from stm32: " + str(received_data))
                    print("time for roundtrip transmission:" + str(latency))
                    print("***********************************************************************")
                    received = True
                    break

            if not received:
                print("Response Package not recevied")
    
        #Test for COLOR packet by Richard, flash examples/minipc/ColorTest.cc
        case Test.COLOR:
            cmd_id = uart.cfg.COLOR_CMD_ID

            data = {'my_color': 'red'}
            print("sending color data:" + str(data['my_color']))
            uart.create_and_send_packet(cmd_id, data)
            send_time = time.time()

            received = False
            #trying to read response for 1 s
            while time.time()-send_time < 1:

                if uart.try_read_one():
                    print(uart.get_circular_buffer())
                if uart.packet_search(): 
                    received_data = uart.get_current_stm32_state()
                    latency = time.time()-send_time
                    print("color data from stm32: " + str(received_data['my_color']))
                    print("time for roundtrip transmission:" + str(latency))
                    print("***********************************************************************")
                    received = True
                    break

            if not received:
                print("Response Package not recevied")
                    
            
        #Latency test by Richard, flash example/minipc/LatencyTest.cc
        case Test.LATENCY:
            # in the packet, rel_yaw is the current time
            i = 0
            cmd_id = uart.cfg.CHASSIS_CMD_ID #it has to be the packet type that has the maximum size
            average_latency = 0

            for i in range(10):
                #sending data
                send_time = time.time()
                data = {'vx': i, 'vy': 0.0, 'vw': 0.0}
                uart.create_and_send_packet(cmd_id, data)
                print("sending packet to stm32...")

                #try to read, 1s timeout
                received=False
                while time.time() - send_time < 1:
                    uart.try_read_one()
                    if uart.packet_search(): 
                        received_data = uart.get_current_stm32_state()
                        latency = time.time()-send_time
                        if(received_data['vx'] == i):
                            print("correct data received")
                            received=True
                        else:
                            print("data received but incorrect")
                        average_latency += latency
                        print("time for roundtrip transmission:" + str(latency))
                        print("***********************************************************************")
                        received = True
                        break
                
                if not received:
                    print("Response Package not recevied")
            average_latency /= 10
            print("average latency is " + str(average_latency))

        # RX/TX test by youhy, flash example/minipc/PingpongTest.cc
        # receive packet from stm32, rel_pitch += 1 then immediately send back
        case Test.PINGPONG:
            i = 0
            cmd_id = uart.cfg.GIMBAL_CMD_ID
            data = {'rel_yaw': 1.0, 'rel_pitch': 2.0, 'mode': 'ST', 'debug_int': 42}
            while True:
                uart.try_read_one()
                # update stm32 status from packet from stm32
                if uart.packet_search():
                    data = uart.get_current_stm32_state()
                    print("from stm32: " + str(data))
                time.sleep(0.005)
                i = i + 1
                # every 0.5 sec, send current status to stm32
                if i % 100 == 0:
                    print("about to send " + str(data))
                    uart.create_and_send_packet(cmd_id, data)
                    i = 0
        
        # RX/TX test by Richard, flash example/minipc/PingPongTestPlus.cc
        # receive packet from stm32, rel_pitch += 1 then immediately send back
        # this ping pong test first trys to send a packet and then attmepts to read the response from stm32 for 10 seconds
        # then send a second packet
        # after that entering ping pong mode 
        # each packet has a ID for differentiating during pingping-ing
        # TODO: this test shows the issue that a response can only be received after the data in circular_buffer is at least 
                    # the maximum size of a packet (STJ_MAX_PACKET_SIZE), so if sending some small packets
                    # they will stay in the circular_buffer waiting to be parsed until new packets are received
        case Test.MODIFIED_PINGPONG:
            i = 0
            cmd_id = uart.cfg.GIMBAL_CMD_ID
            packet_count = 0

            #sending packet the first time
            data = {'rel_yaw': packet_count, 'rel_pitch': 0.0, 'mode': 'ST', 'debug_int': 42}
            print("Sending the first time: ID = " + str(data['rel_yaw']) + " count = " + str(data['rel_pitch']))
            uart.create_and_send_packet(cmd_id, data)

            #attempt to receive packet
            print('attemp to receive and parse response...')
            for i in range(20):
                time.sleep(0.1)
                if uart.try_read_one():
                    print("data waiting at serial port")
                    print("circular buffer:" + str(uart.get_circular_buffer()))
                if uart.packet_search():
                    received_data = uart.get_current_stm32_state()
                    print("from stm32: ID = " + str(received_data['rel_yaw']) + " count = " + str(received_data['rel_pitch']))


            #sending packet the second time
            packet_count += 1
            data = {'rel_yaw': packet_count, 'rel_pitch': 0.0, 'mode': 'ST', 'debug_int': 42}
            print("Sending the second time: ID = " + str(data['rel_yaw']) + " count = " + str(data['rel_pitch']))
            uart.create_and_send_packet(cmd_id, data)

            print('starting ping pong')
            #start ping pong
            while True:
                uart.try_read_one()
                # update stm32 status from packet from stm32
                if uart.packet_search():
                    received_data = uart.get_current_stm32_state()
                    print("from stm32: ID = " + str(received_data['rel_yaw']) + " count = " + str(received_data['rel_pitch']))
                    received_data['rel_pitch'] = received_data['rel_pitch'] + 1
                    uart.create_and_send_packet(cmd_id, received_data)

                time.sleep(0.05)
        case Test.CRC:
            # rate test by Roger modified by Richard, flash example/minipc/StressTestTypeC.cc
            # TODO: currently this test will never receive full 1000 packets but only 998 packets because the last two packets
            # remain in circular buffer and not parsed because its size is not reaching STJ_MAX_PACKET_SIZE
            # NOTE: please reflash or restart program on stm32 every time you want to run this test
            print("Starting packet sending test.")
            for i in range(1000):
                time.sleep(0.005)  # simulate 200Hz
                cmd_id = uart.cfg.GIMBAL_CMD_ID
                data = {'rel_yaw': 1.0, 'rel_pitch': 2.0, 'mode': 'ST', 'debug_int': 42}
                uart.create_and_send_packet(cmd_id, data)

            print("Packet sending test complete.")
            print("You should see the light change from blue to green on type C board.")
            print("When the led turns red the stm32 is sending data.")
            print("Starting packet receiving test.")

            start_time = time.time()
            while time.time() - start_time < 10: #10s timeout
                uart.try_read_one()

                if uart.packet_search():
                    received_data = uart.get_current_stm32_state()
                    # print("from stm32: color = " + str(received_data['my_color']))
                time.sleep(0.001)
            print("Read and parsed %d/1000 packets." % uart.parsed_packet_cnt)
            if uart.parsed_packet_cnt == 1000:
                print("Receiver successfully parsed exactly 100 packets.")
            if uart.parsed_packet_cnt > 1000:
                print("Repeatedly parsed one packet?") 
            print(uart.get_current_stm32_state())
            print("Packet receiving test complete.")
        case Test.TYPE_A:
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
        case _:
            print("Testing Invalid") 
