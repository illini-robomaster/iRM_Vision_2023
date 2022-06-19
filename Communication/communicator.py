import time
import serial
import crc8
import crc16
from config import SERIAL_PORT

serial_port = serial.Serial(
    port=SERIAL_PORT,
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)
# Wait a second to let the port initialize
time.sleep(1)

def create_packet(cmd_id, data, seq):
    '''
    Args:
        cmd_id: bytes, ID for command to send, see "variables" for different cmd
        data: bytes, data to send
        seq: int, n-th packet to send
    Return:
        Bytes of encoded package with format:
            SOF(1 byte) data_len(2 bytes) seq (1 bytes) crc8 (1 bytes) data (x bytes) crc16 (2 bytes)
    '''
    # header
    SOF = b'\xa5'
    data_len = len(data).to_bytes(2,'big')
    seq = seq.to_bytes(1,'big')
    hash = crc8.crc8() #crc8
    hash.update(SOF+data_len+seq)
    crc_header = hash.digest()

    #tail
    crc_data = crc16.crc16xmodem(data).to_bytes(2,'big')

    pkt = SOF+data_len+seq+crc_header+cmd_id+data+crc_data
    return pkt

def send_packet(pkt, serial_port):
    serial_port.write(pkt)

if __name__ =='__main__':
    try:
        cmd_id = b'\xde\xad'
        data = 0xffff*b'\xAA'
        pkt = create_packet(cmd_id,data,0)
        serial_port.write(pkt)
        while True:
            if serial_port.inWaiting() > 0:
                data = serial_port.read()
                print(data)
    except:
        print("Falied to write")


class Communicator:
    def __init__(self):
        pass

    def move(self, yaw_offset, pitch_offset):
        pass
