# iRM Communication Protocol

This documents the communication protocol design for exchanging information between
onboard Nvidia Jetson computers and the STM32 control board.

For implementation on the STM32 side, please refer to the example [here](https://github.com/illini-robomaster/iRM_Embedded_2023/tree/main/examples/minipc).

For implementation on the Jetson side, please refer to the main communicator [here](../Communication/communicator.py).

# Protocol between Jetson and STM32

## Packet Struct

| Name         | Content                            | Size          |
|--------------|------------------------------------|---------------|
| HEADER       | fixed 2 ASCII char ('ST')          | 2 bytes       |
| SEQ_NUM      | uint16_t counter                   | 2 bytes       |
| DATA_LEN     | uint8_t length of the data section | 1 byte        |
| CMD_ID       | uint8_t identifier                 | 1 byte        |
| DATA         | struct with fixed length           | 1 - 12 bytes  |
| CRC_CHECKSUM | uint8_t checksum                   | 1 byte        |
| TAIL         | fixed 2 ASCII char ('ED')          | 2 bytes       |
| TOTAL        |                                    | 10 - 21 bytes |

## Detailed explanations

Note: the memory layout of STM32 is **little-endian**. Hence when integers that require
decoding are sent, they need to be encoded in little endianness. However, for strings
(e.g., the HEADER ASCII and PACK_END ASCII), they do not need to be reversed since array
layouts are not affected by endianness.

### HEADER

2 ASCII chars This is the start of the packet denoted by ASCII characters 'ST'.


### SEQ_NUM

uint16_t counter. This is a counter that is incremented by 1 every time a packet is sent. Restart from 0 when overflow.

### DATA_LEN

uint8_t length. The length of the data section (see below). Left here for future extension. Currently for all CMD_ID, the length of data is fixed and known. So this value is set to 0.

### CMD_ID

uint8_t identifier.

| Name           | ID   | Data Length | Description                      |
|----------------|------|-------------|----------------------------------|
| GIMBAL_CMD_ID  | 0x00 | 12          | Movement of the gimbal           |
| COLOR_CMD_ID   | 0x01 | 1           | My color, red is 0 and blue is 1 |
| CHASSIS_CMD_ID | 0x02 | 12          | Movement of the chassis          |

Packet length = Data length + 9 bytes _(Head length + Tail length + Checksum length)_

### DATA

The followings are current data structs:

---

#### gimbal_data_t

| Name       | Content                    | Size    |
|------------|----------------------------|---------|
| rel_yaw    | float32, relative yaw      | 4 bytes |
| rel_pitch  | float32, relative pitch    | 4 bytes |
| mode       | uint8_t, autoaim mode      | 1 byte  |
| debug_int  | uint8_t, debug int         | 1 byte  |

**mode**: Different modes for autoaim. Current options are 'ST' (0) and 'MY' (1), which stands for "Search Target" and "Move Yoke".

- **ST == 0**: The "Search Target" mode means either 1) the AutoAim algorithm is not engaged
    or 2) the AutoAim algorithm is engaged but the target is not found. In the former case, the
    STM32 board should simply ignore input from the Jetson board. In the latter case, the STM32
    should hold pitch to the horizontal position, and rotate yaw to search for the target.
- **MY == 1**: The "Move Yoke" mode means the AutoAim algorithm is engaged and the target is found.
    In this case, the relative pitch/yaw angles are sent to the STM32 board, and the gimbal
    should move relatively to the current position. (**NOTE**: filtering to compensate for
    latency and smoothness needs to be implemented on the STM32 side due to real-time access
    to gimbal data.)

**debug_int**: A int for debug for user.

---

#### color_data_t

| Name     | Content                    | Size   |
|----------|----------------------------|--------|
| my_color | uint8_t, color of our team | 1 byte |

**my_color**: RED is 0; BLUE is one

---

#### chassis_data_t

| Name | Content                                       | Size    |
|------|-----------------------------------------------|---------|
| vx   | float32, velocity to the front of the chassis | 4 bytes |
| vy   | float32, velocity to the left of the chassis  | 4 bytes |
| vw   | float32, counterclockwise angular velocity    | 4 bytes |

---

### CRC_CHECKSUM

uint8_t checksum. This is the CRC checksum of the packet. The CRC standard used
is the MAXIM_DOW standard. The CRC checksum is calculated on the packet contents BEFORE the
PACK_END (i.e., CRC is computed for the first (PACKET_LEN - 3) bytes up to end to REL_PITCH).

### PACK_END

2 ASCII chars. This is the end of the packet denoted by ASCII characters 'ED'.
