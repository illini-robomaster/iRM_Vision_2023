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
import serial
import logging
import argparse
import threading
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

# Add loggers here
loggers = [logger := logging.getLogger(__name__),
           c_logger := logging.getLogger('Communication.communicator')]


# UART > 0, USB <= 0
class DeviceType(IntEnum):
    UART = 1  # serialless uart, also acts to `type' others
    BRD = 2  # board

    USB = 0  # serialless usb, also acts to `type' others
    SPM = -1  # spacemouse
    ARM = -2  # arm


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

def dname(intenum: DeviceType) -> str:
    return intenum_to_name[intenum]

# This is expected to be thread-safe.
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
ap.add_argument('-s', '--skip-tests',
                action='store_true',
                help='alias for --test 0')
ap.add_argument('--test-only',
                action='store_true',
                help='exit after completing tests')
ap.add_argument('--exit-on-detach',
                action='store_true',
                help='exit if no devices are attatched')
ap.add_argument('-V', '--version',
                action='version',
                version='%(prog)s 0.1')

in_use = AtomicList()

def is_uart(dev_type: DeviceType) -> bool:
    return dev_type >= UART

def is_usb(dev_type: DeviceType) -> bool:
    return dev_type <= USB

def get_device(dev_type: DeviceType) -> 'Communicator':
    return serial_devices[dev_type]

def assign_device(dev_type: DeviceType, device: 'Communicator') -> None:
    serial_devices[dev_type] = device

def deassign_device(dev_type: DeviceType) -> None:
    dev_path = get_device(dev_type).serial_port.port
    assign_device(dev_type, None)
    in_use.remove(dev_path)

#
# For unwrapped packets
#
def write_packet(
        dev_type: DeviceType, cmd_id: hex, data: dict) -> bytes:
    dev = get_device(dev_type)
    return dev.create_packet(cmd_id, data)

def send_packet(dev_type: DeviceType, packet: bytes) -> None:
    dev = get_device(dev_type)
    return dev.send_packet(packet)

def write_send_packet(
        dev_type: DeviceType, cmd_id: hex, data: dict) -> None:
    dev = get_device(dev_type)
    return dev.create_and_send_packet(cmd_id, data)

def receive_packet(dev_type=UART) -> Optional[dict]:
    dev = get_device(dev_type)
    if dev_type in (UART, BRD):
        return dev.get_current_stm32_state()
    elif dev_type in (USB, SPM, ARM):
        return None
    else:
        logger.error(f'Failed to receive packet: not a serial device: {dev}')

# e.g. ''.join( oct_bin_to_str_rev(0o12) ) == '0101'
def oct_bin_to_str_rev(o: oct) -> str:
    return reversed(str(bin(o))[2:])  # Remove `0b'

def startup_tests(
        testable: List[Tuple[Callable, DeviceType]], verb=0o10) -> None:
    # Convert octal to binary representation and run tests.
    # i.e. oct(0o16) => '0111' runs test_board_{latency,pingpong,crc}.
    lst = oct_bin_to_str_rev(verb)
    selected_tests = [t
                      for b, t in zip(lst, testable)
                      if b == '1']
    return [action(get_device(dev_type))
            for action, dev_type in selected_tests]

def get_communicator(
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
        print(f'=> {dname(d_t):<15} %s' %
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
        print(f'==> Please {GREEN}_plug_{RESET} the {dname(d_t)}.')
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
        new_dev_lst += [get_communicator(d_t, config, serial_dev_path=srl_dev_pth)]
    print('\n==> New serial assignments:')
    for d_t, dev in zip([t for t, _ in d_t_dev_lst], new_dev_lst):
        print(f'=> {dname(d_t):<15} %s' %
              (f'{YELLOW}(no serial port assigned){RESET}'
               if not dev.is_valid()
               else dev.serial_port))
    print(f'\n{GREEN}{BOLD}==> Do you want to continue with this '
          f'configuration?{NOBOLD}{RESET}')
    perform_plug_test(d_t_dev_lst) \
            if input("=> ('NO' to retry) ") == 'NO' \
            else None
    print()

    return new_dev_lst

def set_up_loggers(
        loggers: List[logging.Logger], verbosity) -> None:
    ch = logging.StreamHandler()
    ch.setFormatter(cl.ColorFormatter())
    for lgr in loggers:
        lgr.handlers.clear()
        lgr.addHandler(ch)
        lgr.setLevel(verbosity or 'INFO')

def set_up_devices(*args: Tuple[DeviceType, str],
                   plug_test: bool) -> Optional['Communicator']:
    # XXX: Currently a generator. Change if used more than once.
    dev_pth_zip = args

    ret_list = []  # Of serial devices or None
    for dev, pth in dev_pth_zip:
        logger.info(f'Setting up {dname(dev)} (given path {pth})')
        if pth is not None:
            ret_list += [get_communicator(dev, config, serial_dev_path=pth)]
        elif not plug_test:
            ret_list += [get_communicator(dev, config)]
        else:
            ret_list += [None]

    return ret_list

def main(pa) -> None:
    #
    # Run self check tests
    #
    if not pa.skip_tests:
        testable: List[Tuple[Callable, DeviceType]] = [
                (communicator.test_board_typea, BRD),     # 0o1
                (communicator.test_board_crc, BRD),       # 0o2
                (communicator.test_board_pingpong, BRD),  # 0o4
                (communicator.test_board_latency, BRD),   # 0o10
                ]
        startup_tests(testable, int(pa.test, 8))
    if pa.test_only:
        exit(0)
    #
    # Begin listening
    #
    listening_devices: List[Tuple[DeviceType, serial.Serial]] = [
            (dev_type, dev)
            for dev_type, dev in serial_devices.items()
            if dev is not None
            if dev.is_valid()]

    if not listening_devices and pa.exit_on_detach:
        logger.error('No devices attatched: exiting...')
        exit(1)
    
    for dev_type, dev in listening_devices:
            dev.start_listening()
            print(f'{intenum_to_name[dev_type]} listening on '
                  f'{dev.serial_port}')

    def brd_sender():
        while True:
            # Get instructions from some atomic structure
            #write_send_packet(cmd_id, data, BRD)
            time.sleep(1/60)
    def brd_listener():
        while True:
            time.sleep(1/120)
    #def spm_sender():
    #    while True:
    #        time.sleep(1/60)
    def spm_listener():
        while True:
            time.sleep(1/120)
    #def arm_sender():
    #    while True:
    #        time.sleep(1/60)
    def arm_listener():
        while True:
            time.sleep(1/120)
    #
    # Main loop
    #
    brd_sender_thread = threading.Thread(target=brd_sender)
    brd_listener_thread = threading.Thread(target=brd_listener)
    #spm_sender_thread = threading.Thread(target=spm_sender)
    spm_listener_thread = threading.Thread(target=spm_listener)
    #arm_sender_thread = threading.Thread(target=arm_sender)
    arm_listener_thread = threading.Thread(target=arm_listener)

    j = 0
    while True:
        j += 1
        #cmd_id = config.SELFCHECK_CMD_ID
        #data = {'mode': 'EC', 'debug_int': j}
        cmd_id = config.ARM_CMD_ID
        data = {
            'floats': {'float0': 0.0,
                       'float1': 1.0,
                       'float2': 2.0,
                       'float3': 3.0,
                       'float4': 4.0,
                       'float5': 5.0,
            }
        }
        # TODO: Allow reconnecting, i.e. with id requests
        try:
            if get_device(BRD) is not None:
                write_send_packet(BRD, cmd_id, data)
                print(receive_packet(BRD))
            else:
                print('Nothing connected')

        except serial.serialutil.SerialException:
            logger.error(f'Lost connection to {dname(BRD)}!')
            deassign_device(BRD)

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
    #
    # Set up logging.
    #
    set_up_loggers(loggers, parsed_args.verbosity)
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
    # Plug test
    # *_dev :: UARTCommunicator
    # *_dev.serial_port :: serial.Serial
    # *_dev.serial_port.port :: Maybe 'path'
    if parsed_args.plug_test:
        brd_dev, spm_dev, arm_dev = \
            perform_plug_test((BRD, brd_dev),
                              (SPM, spm_dev),
                              (ARM, arm_dev),)
    assign_device(BRD, brd_dev)
    assign_device(SPM, spm_dev)
    assign_device(ARM, arm_dev)

    ###########################
    #  End argument handling  #
    ###########################
    main(parsed_args)
