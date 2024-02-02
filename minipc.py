#!/usr/bin/env python3
""" Extending this script:
    When making modifications, change:
        0) The configuration file (config.py)

        1) To add a new logger, modify:
        1.1) loggers  = [

        2) The identification scheme
        2.1) class DeviceType(IntEnum):
        2.2) intenum_to_name = {
        2.3) if parsed_args.plug_test:
        2.4) serial_devices[.*\] = 
        2.5) serial_devices = {

        3) Argparse

        4) Sending and receiving packets

    Device (`dev') refers to a *Communicator instance.
"""
import sys
import time
import logging
import argparse
from enum import IntEnum
from typing import Optional, Union, \
                   Sequence, Tuple, List, \
                   Set, \
                   Mapping, Dict, \
                   Callable
from subprocess import run as sp_run

import config  # TODO: support multiple config files/config.ConfigA etc

from Utils import color_logging as cl
from Utils.ansi import *
from Utils.atomiclist import AtomicList

from Communication import communicator
from Communication.communicator import UARTCommunicator, USBCommunicator
from Communication.packet import UARTPacket, USBPacket

# Add loggers here
loggers = [logger := logging.getLogger(__name__),
           c_logger := logging.getLogger('Communication.communicator')]


# UART > 0, USB <= 0
class DeviceType(IntEnum):
    UART = 1  # serialless uart, also acts to `type' others
    BRD = 2  # board
    SPM = 3  # spacemouse

    USB = 0  # serialless usb, also acts to `type' others
    ARM = -1  # arm


UART = DeviceType.UART
USB = DeviceType.USB
BRD = DeviceType.BRD
SPM = DeviceType.SPM
ARM = DeviceType.ARM

intenum_to_name = {
    UART: 'UART',
    USB: 'USB',
    BRD: 'BOARD',
    SPM: 'SPACEMOUSE',
    ARM: 'ARM',
}

def get_name(intenum: DeviceType) -> str:
    return intenum_to_name[intenum]

# Muting data in an (Int)Enum is not expected behavior, so:
serial_devices: Dict[DeviceType, 'Communicator'] = {
    UART: UARTCommunicator(config, serial_dev_path=False, warn=False),
    USB: None,  # XXX: Replace when USBCommunicator is ready
    BRD: None,
    SPM: None,
    ARM: None,
}

ap = argparse.ArgumentParser(description='Control script for minipc',
                             prog='minipc.py',
                             epilog='test_board_latency:0o10, '
                             'test_board_pingpong:0o4, '
                             'test_board_crc:0o2, '
                             'test_board_typea:0o1')
# TODO: handle numeric verbosity
ap.add_argument('-v', '--verbosity', '--verbose',
                action='store',
                default='WARNING',
                nargs='?',
                help='DEBUG, INFO, WARNING, ERROR, or CRITICAL')
ap.add_argument('-p', '--plug-test',
                action='store_true',
                default=False,
                help='assign serial ports interactively [recommended]')
ap.add_argument('-B', '--board-port',
                action='store',
                default=None)
ap.add_argument('-S', '--spacemouse-port',
                action='store',
                default=None)
ap.add_argument('-A', '--arm-port',
                action='store',
                default=None)
ap.add_argument('-t', '--test', action='store', default='0o10',
                help='takes an octal value; values below')
ap.add_argument('--skip-tests',
                action='store_true',
                help='alias for --test 0')
ap.add_argument('--test-only',
                action='store_true',
                help='exit after completing tests')
ap.add_argument('-V', '--version',
                action='version',
                version='%(prog)s 0.1')

# 赛博跳大神
in_use = AtomicList()

def is_uart(dev_type: DeviceType) -> bool:
    return dev_type >= UART

def is_usb(dev_type: DeviceType) -> bool:
    return dev_type <= USB

#
# For unwrapped packets
#
def write_packet_uw(
        cmd_id: hex, data: dict, dev_type=UART) -> bytes:
    if is_uart(dev_type):
        uart = serial_devices[dev_type]
        unwrapped_packet = uart.create_packet(cmd_id, data)
    elif is_usb(dev_type):
        usb = serial_devices[dev_type]
        unwrapped_packet = None
    else:
        logger.error(f'Failed to encode: not a serial device: {dev_type}')
    return unwrapped_packet

def send_packet_uw(packet: bytes, dev_type=UART) -> None:
    if is_uart(dev_type):
        uart = serial_devices[dev_type]
    elif is_usb(dev_type):
        usb = serial_devices[dev_type]
    else:
        logger.error(f'Failed to send: not a serial device: {dev_type}')

def write_send_packet_uw(
        cmd_id: hex, data: dict, dev_type=UART) -> bytes:
    unwrapped_packet = write_packet_uw(cmd_id, data, dev_type)
    send_packet_uw(unwrapped_packet)
    return unwrapped_packet
#
# For wrapped packets
#
def write_packet(
        cmd_id: hex, data: dict, dev_type=UART, dump=False) -> 'Packet':
    if is_uart(dev_type):
        uart = serial_devices[dev_type]
        packet = UARTPacket(cmd_id, data, uart)
    elif is_usb(dev_type):
        usb = serial_devices[dev_type]
        packet = None
    else:
        logger.error(f'Failed to encode Packet: not a serial device: {dev_type}')

    if packet is not None and dump:
        logging.debug(packet.dump_self(dump))

    return packet

def send_packet(
        packet: 'Packet', dev_type=UART, dump=False) -> None:
    if dump:
        logging.debug(packet.dump_self(dump))

    if is_uart(dev_type):
        uart = serial_devices[dev_type]
        uart.send_packet(packet.contents)
    elif is_usb(dev_type):
        usb = serial_devices[dev_type]
        #usb.send_packet(packet.contents)
    else:
        logger.error(f'Failed to send Packet: not a serial device: {dev_type}')

def write_send_packet(
        cmd_id: hex, data: dict, dev_type=UART) -> bytes:
    unwrapped_packet = write_packet_uw(cmd_id, data, dev_type)
    send_packet_uw(unwrapped_packet)
    return unwrapped_packet

def receive_packet(dev_type=UART) -> Optional[dict]:
    if is_uart(dev_type):
        uart = serial_devices[dev_type]
        if dev_type in (UART, BRD):
            return uart.get_current_stm32_state()
        else:
            return None
    elif is_usb(dev_type):
        return None
    else:
        logger.error(f'Failed to receive packet: not a serial device: {dev}')

def oct_bin_to_str(o: oct) -> str:
    return str(bin(o))[2:]  # Remove `0b'

# XXX: currently only latency, crc modified to work
def startup_tests(
        testable: List[Tuple[Callable, DeviceType]], verb=0o10) -> None:
    # Convert octal to binary representation and run tests.
    # i.e. oct(0o16) => '001110' runs test_board_{latency,pingpong,crc}.
    lst = oct_bin_to_str(verb)
    selected_tests = [t
                      for b, t in zip(lst, testable)
                      if b == '1']
    return [action(serial_devices[dev_type])
            for action, dev_type in selected_tests]

def assign_device(
        dev_type: DeviceType, *args, **kwargs) -> 'Communicator':
    if is_uart(dev_type):
        return UARTCommunicator(in_use=in_use, *args, **kwargs)
    elif is_usb(dev_type):
        # XXX: Replace when USBCommunicator is ready
        return UARTCommunicator(in_use=in_use, *args, **kwargs)
    else:
        logger.error(f'Failed to encode: not a serial device: {dev}')

# TODO: Add `udevadm` integration?
def perform_plug_test(
        *args: Tuple[DeviceType, 'Communicator']) -> List['Communicator']:
    # [d]ev[_t]ype, [dev]ice
    d_t_dev_lst: List[Tuple[DeviceType, 'Communicator']] = [*args]
    print('==> Current serial assignments:')
    for d_t, dev in d_t_dev_lst:
        print(f'=> {get_name(d_t):<15} %s' %
              f'{RED}(no device){RESET}'
              if not dev
              else dev.serial_port)
    # If all serial ports are assigned,
    if None not in (dev_l := [d for _, d in d_t_dev_lst]):
        print('==> Nothing left to assign.')
        # return the devices we received.
        return dev_l

    logger.info(f'Not all devices were assigned, continuing on to plug test.')

    new_dev_lst = []  # To be returned
    blacklisted_ports: Set['path'] = {None}  # Ports to ignore
    print(f'==> Please {YELLOW}_unplug_{RESET} all devices to be assigned.')
    input('=> (enter to continue) ')
    for d_t, dev in d_t_dev_lst:
        blacklisted_ports.add(
            *UARTCommunicator.list_uart_device_paths(),
            # XXX: Replace when USBCommunicator is ready
            # *USBCommunicator.list_usb_device_paths())
            )
        print(f'=> Possible ports in blacklist: {blacklisted_ports}')
        print(f'==> Please {GREEN}_plug_{RESET} the {get_name(d_t)}.')
        input('=> (enter to continue) ')
        new_ports: List['path'] = [
            *UARTCommunicator.list_uart_device_paths(),
            # XXX: Replace when USBCommunicator is ready
            # *USBCommunicator.list_usb_device_paths())
        ]
        new_ports = [pth
                     for pth in new_ports
                     if pth not in blacklisted_ports]
        if not new_ports:
            print(f'==> {YELLOW}No new devices found, '
                  f'assigning to serialless device.{RESET}')
            srl_dev_pth = False
        elif (_len := len(new_ports)) > 1:
            print(f'==> {YELLOW}More than one device found.{YELLOW}')
            print('==> It may be helpful to enter a number:')
            print('\n'.join([f'{i+1}: {d}']
                            for i, d in enumerate(new_ports)))
            _ind = input(f'=> (1 through {_len}) ')
            print(f'=> {GREEN}Using device: {new_ports[_ind]}{RESET}')
            srl_dev_pth = new_ports[_ind]
        else:
            print(f'=> {GREEN}Using device: {new_ports[0]}{RESET}')
            srl_dev_pth = new_ports[0]
        new_dev_lst += [assign_device(d_t, config, serial_dev_path=srl_dev_pth)]
    print('\n==> New serial assignments:')
    for d_t, dev in zip([t for t, _ in d_t_dev_lst], new_dev_lst):
        print(f'=> {get_name(d_t):<15} %s' %
              (f'{YELLOW}(no serial port assigned){RESET}'
               if not dev.serial_port
               else dev.serial_port))
    print(f'\n{GREEN}{BOLD}==> Do you want to continue with this '
          f'configuration?{NOBOLD}{RESET}')
    perform_plug_test(d_t_dev_lst) \
            if input("=> ('NO' to retry) ") == 'NO' \
            else None
    print()

    return new_dev_lst

def set_up_loggers(
        loggers: List[logging.Logger]) -> None:
    ch = logging.StreamHandler()
    ch.setFormatter(cl.ColorFormatter())
    for lgr in loggers:
        lgr.handlers.clear()
        lgr.addHandler(ch)
        lgr.setLevel(parsed_args.verbosity or 'INFO')

def set_up_devices(*args: Tuple[DeviceType, str],
                   plug_test: bool) -> Optional['Communicator']:
    # XXX: Currently a generator. Change if used more than once.
    dev_pth_zip = args

    ret_list = []  # Of serial devices or None
    for dev, pth in dev_pth_zip:
        logger.info(f'Setting up {get_name(dev)} (given path {pth})')
        if pth is not None:
            ret_list += [assign_device(
                dev, config, serial_dev_path=pth)]
        elif not plug_test:
            ret_list += [assign_device(dev, config)]
        else:
            ret_list += [None]

    return ret_list

def main(args) -> None:
    #
    # Run self check tests
    #
    testable: List[Tuple[Callable, DeviceType]] = [
            (communicator.test_board_latency, BRD),   # 0o10
            (communicator.test_board_pingpong, BRD),  # 0o4
            (communicator.test_board_crc, BRD),       # 0o2
            (communicator.test_board_typea, BRD),     # 0o1
            ]
    startup_tests(testable, int(args.test, 8))
    if parsed_args.test_only:
        exit(0)

    #
    # Begin listening
    #
    listening_devices: List[Tuple[DeviceType, serial.Serial]] = [
            (dev_type, dev)
            for dev_type, dev in serial_devices.items()
            if dev is not None
            if dev.serial_port is not None]

    if not listening_devices:
        logger.error('No devices attatched: exiting...')
        exit(1)
    
    for dev_type, dev in listening_devices:
            dev.start_listening()
            print(f'{intenum_to_name[dev_type]} listening on '
                  f'{dev.serial_port}')

    #
    # Main loop
    #
    # TODO: Everything
    j = 0
    while True:
        j += 1
        cmd_id = config.SELFCHECK_CMD_ID
        data = {'rel_yaw': j, 'rel_pitch': j,
                'mode': 'EC', 'debug_int': j}
        # packet = write_packet(cmd_id, data, BRD)
        # send_packet(packet)

        # print(receive_packet())
        time.sleep(1)

if __name__ == '__main__':
    # Remove first arg if called with python.
    _sl = slice(2, None) if 'python' in sys.argv[0] \
                         else slice(1, None)
    """ Items in parsed_args:
    parsed_args.verbosity: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
                                                      (default WARNING,
                                                       w/o arg INFO)
    parsed_args.plug_test: boolean
    parsed_args.board_port: '/path/to/some/port'      (d. None)
    parsed_args.spacemouse_port: '/path/to/some/port' (d. None)
    parsed_args.arm_port: '/path/to/some/port'        (d. None)
    parsed_args.test: 'octal'                         (d. '0o10')
    parsed_args.skip_tests: boolean
    parsed_args.test_only: boolean
    """
    parsed_args = ap.parse_args(sys.argv[_sl])
    #############################
    #  Handle parsed arguments  #
    #############################
    #  TODO: organize
    #
    # Set up logging.
    #
    set_up_loggers(loggers)
    logger.debug(parsed_args)

    #
    # Set up serial devices.
    #
    # *_dev :: Maybe UARTCommunicator
    brd_dev, spm_dev, arm_dev = \
        set_up_devices((BRD, parsed_args.board_port),
                       (SPM, parsed_args.spacemouse_port),
                       (ARM, parsed_args.arm_port),
                       plug_test=parsed_args.plug_test)

    #
    # Plug test
    #
    # *_dev :: UARTCommunicator
    # *_dev.serial_port :: serial.Serial
    # *_dev.serial_port.port :: Maybe 'path'
    if parsed_args.plug_test:
        brd_dev, spm_dev, arm_dev = \
            perform_plug_test((BRD, brd_dev),
                              (SPM, spm_dev),
                              (ARM, arm_dev),)

    serial_devices[BRD] = brd_dev
    serial_devices[SPM] = spm_dev
    serial_devices[ARM] = arm_dev

    ###########################
    #  End argument handling  #
    ###########################
    main(parsed_args)
