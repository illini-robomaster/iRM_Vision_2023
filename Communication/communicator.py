import time
import serial
import crc8
import crc16

import config

# USB is for TTL-only device
USB_PREFIX = "/dev/ttyUSB"
# ACM is for st-link/TTL device
ACM_PREFIX = "/dev/ttyACM"

success_flag = False

for prefix in [USB_PREFIX, ACM_PREFIX]:
    if success_flag: break
    for i in range(5):
        if success_flag: break
        try:
            serial_port = serial.Serial(
                port=prefix + str(i),
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            success_flag = True
            break # if succeed, break
        except serial.serialutil.SerialException:
            serial_port = None

# Wait a second to let the port initialize
time.sleep(1)

def create_packet(header, seq_num, yaw_offset, pitch_offset):
    assert header in [config.SEARCH_TARGET, config.MOVE_YOKE]
    packet = header
    assert seq_num >= 0 and seq_num < 2**32 - 1 # uint32
    packet += seq_num.to_bytes(4, 'big')
    # YAW/PITCH offset should not be too high
    assert yaw_offset >= -config.RGBD_CAMERA.YAW_FOV_HALF and yaw_offset <= config.RGBD_CAMERA.YAW_FOV_HALF
    assert pitch_offset >= -config.RGBD_CAMERA.PITCH_FOV_HALF and pitch_offset <= config.RGBD_CAMERA.PITCH_FOV_HALF
    discrete_yaw_offset = int(yaw_offset * 100000)
    discrete_pitch_offset = int(pitch_offset * 100000)
    packet += (discrete_yaw_offset & 0xFFFFFFFF).to_bytes(4, 'big')
    packet += (discrete_pitch_offset & 0xFFFFFFFF).to_bytes(4, 'big')
    # ENDING
    packet += config.PACK_END
    return packet

def create_packet_w_crc(cmd_id, data, seq):
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

# Testing code
if __name__ =='__main__':
    assert serial_port is not None, "No serial device found; check root priviledge and USB devices"
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
