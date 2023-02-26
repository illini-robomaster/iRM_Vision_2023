# iRM Communication Protocol

This documents the communication protocol design for exchanging information between
onboard Nvidia Jetson computers and the STM32 control board.

For implementation on the STM32 side, please refer to the example [here](https://github.com/illini-robomaster/iRM_Embedded_2023/tree/main/examples/minipc).

For implementation on the Jetson side, please refer to the main communicator [here](../Communication/communicator.py).

# Protocol from Jetson to STM32

## Packet Struct

| Name          | Content                   | Size    |
| ------------- | ------------------------- | ------- |
| HEADER        | 2 ASCII char              | 2 bytes |
| SEQ_NUM       | uint32_t counter          | 4 bytes |
| REL_YAW       | int32_t discretized float | 4 bytes |
| REL_PITCH     | int32_t discretized float | 4 bytes |
| CRC_CHECKSUM  | uint8_t checksum          | 1 bytes |
| PACK_END      | 2 ASCII char              | 2 bytes |
| TOTAL         |                           | 17 bytes|

## Detailed explanations

Note: the memory layout of STM32 is **little-endian**. Hence when integers that require
decoding are sent, they need to be encoded in little endianness. However, for strings
(e.g., the HEADER ASCII and PACK_END ASCII), they do not need to be reversed since array
layouts are not affected by endianness.

### HEADER
2 ASCII chars. Current options are 'ST' and 'MY', which stands for "Search Target"
and "Move Yoke".
- **ST**: The "Search Target" mode means either 1) the AutoAim algorithm is not engaged
    or 2) the AutoAim algorithm is engaged but the target is not found. In the former case, the
    STM32 board should simply ignore input from the Jetson board. In the latter case, the STM32
    should hold pitch to the horizontal position, and rotate yaw to search for the target.
- **MY**: The "Move Yoke" mode means the AutoAim algorithm is engaged and the target is found.
    In this case, the relative pitch/yaw angles are sent to the STM32 board, and the gimbal
    should move relatively to the current position. (**NOTE**: filtering to compensate for
    latency and smoothness needs to be implemented on the STM32 side due to real-time access
    to gimbal data.)

### SEQ_NUM

uint32_t counter. This is a counter that is incremented by 1 every time a packet is sent.

### REL_YAW

int32_t discretized float. This is the relative yaw angle to the current yaw angle. It is
discretized by multiplying the float by 1e+6 and then casting to int32_t. For example, if the
relative yaw angle is 0.123456, then the int32_t value should be 123456.

### REL_PITCH

int32_t discretized float. This is the relative pitch angle to the current pitch angle.
It is discretized by multiplying the float by 1e+6 and then casting to int32_t. For example, if
the relative pitch angle is 1, then the int32_t value should be 1000000.

### CRC_CHECKSUM

uint8_t checksum. This is the CRC checksum of the packet. The CRC standard used
is the MAXIM_DOW standard. The CRC checksum is calculated on the packet contents BEFORE the
PACK_END (i.e., CRC is computed for the first 14 bytes up to end to REL_PITCH).

### PACK_END

2 ASCII chars. This is the end of the packet denoted by ASCII characters 'ED'.

# Protocol from STM32 to Jetson

TODO: to be implemented. The first implementation should tell Jetson what is the color of
the armor of our own robots. However it should be implemented in a generic manner so that
subsequent implementations can easily add other useful information.
