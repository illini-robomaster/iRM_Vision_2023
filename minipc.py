#!/usr/bin/env python3
""" Will document later :)
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

import config
from Utils import color_logging as cl
from Utils.ansi import *
from Utils.matchall import MatchAll
from Utils.atomiclist import AtomicList
from Communication import communicator
from Communication.communicator import Communicator, \
                                       UARTCommunicator, USBCommunicator, \
                                       StateDict

_m = MatchAll()

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
ap.add_argument('-t', '--test', action='store', default='0o10',
                help='takes an octal value; values below')
ap.add_argument('-s', '--skip-tests',
                action='store_true',
                help='alias for --test 0')
ap.add_argument('--test-only',
                action='store_true',
                help='exit after completing tests')
ap.add_argument('-V', '--version',
                action='version',
                version='%(prog)s 0.1')
ap.add_argument('--dummy',
                action='store_true',
                help=argparse.SUPPRESS)

# Add loggers here
loggers = [logger := logging.getLogger(__name__),
           c_logger := logging.getLogger('Communication.communicator')]


# UART > 0, USB <= 0
class DeviceType(IntEnum):
    UART = 1  # portless uart
    BRD = 2  # board

    USB = 0  # portless usb
    SPM = -1  # spacemouse
    ARM = -2  # small arm

UART = DeviceType.UART
USB = DeviceType.USB
BRD = DeviceType.BRD
SPM = DeviceType.SPM
ARM = DeviceType.ARM

# This is expected to be thread-safe.
# Muting data in an (Int)Enum is not expected behavior, so:
serial_devices: Dict[DeviceType, Communicator] = {
    UART: UARTCommunicator(config, serial_dev_path=False, warn=False),
    USB: UARTCommunicator(config, serial_dev_path=False, warn=False),  # XXX: Replace when USBCommunicator is ready
    BRD: None,
    SPM: None,
    ARM: None,
}

intenum_to_name = {
    UART: 'UART',
    USB: 'USB',
    BRD: 'BOARD',
    SPM: 'SPACEMOUSE',
    ARM: 'ARM',
}

in_use: List['path'] = AtomicList()
# XXX: Change names or reimplement? Only send_queue actually acts as a queue.
id_queue: List[DeviceType] = AtomicList()
send_queue: List[Tuple[DeviceType, bytes]] = AtomicList()
listen_queue: List[Tuple[DeviceType, Communicator]] = AtomicList()
unified_state = StateDict()

def dname(intenum: DeviceType) -> str:
    return intenum_to_name[intenum]

def get_device(dev_type: DeviceType) -> Communicator:
    return serial_devices[dev_type]

def is_uart(dev_type: DeviceType) -> bool:
    return dev_type >= UART

def is_usb(dev_type: DeviceType) -> bool:
    return dev_type <= USB

# Remove an entry from `in_use'
def delist_device(device: Communicator) -> None:
    dev_path = device.get_port()
    if dev_path:
        logger.debug(f'Freeing {dev_path}.')
        in_use.remove(dev_path)

def get_communicator(
        dev_type: DeviceType, *args, **kwargs) -> Communicator:
    if is_uart(dev_type):
        return UARTCommunicator(in_use=in_use, *args, **kwargs)
    elif is_usb(dev_type):
        # XXX: Replace when USBCommunicator is ready
        return UARTCommunicator(in_use=in_use, *args, **kwargs)


def set_up_loggers(
        loggers: List[logging.Logger], verbosity) -> None:
    ch = logging.StreamHandler()
    ch.setFormatter(cl.ColorFormatter())
    for lgr in loggers:
        lgr.handlers.clear()
        lgr.addHandler(ch)
        lgr.setLevel(verbosity or 'INFO')


#
# XXX: Only _identifier is expected to work with anonymous devices.
# Only use `create_packet` and `push_to_send_queue` outside of the three
# threads (_identifier, _listener, _sender), everything else is for internal
# use only.
# 
def _create_packet_dev(
        device: Communicator, cmd_id: hex, data: dict) -> bytes:
    return device.create_packet(cmd_id, data)

def _send_packet_dev(device: Communicator, packet: bytes) -> None:
    return device.send_packet(packet)

def _create_and_send_packet_dev(
        device: Communicator, cmd_id: hex, data: dict) -> None:
    return device.create_and_send_packet(cmd_id, data)

def _read_packet_dev(device: Communicator) -> dict:
    return device.read_out()

def _receive_uart_packet_dev(device: Communicator) -> None:
    device.try_read_one()
    device.packet_search()

def _send_packet(dev_type: DeviceType, packet: bytes) -> None:
    device = get_device(dev_type)
    return _send_packet_dev(device, packet)

def _create_and_send_packet(
        dev_type: DeviceType, cmd_id: hex, data: dict) -> None:
    device = get_device(dev_type)
    return _create_and_send_packet_dev(device, cmd_id, data)

def _read_packet(dev_type: DeviceType) -> dict:
    device = get_device(dev_type)
    return _read_packet_dev(device)

def create_packet(
        dev_type: DeviceType, cmd_id: hex, data: dict) -> bytes:
    device = get_device(dev_type)
    return _create_packet_dev(device, cmd_id, data)

def push_to_send_queue(dev_type: DeviceType, packet: bytes) -> None:
    global send_queue
    send_queue += [(dev_type, packet)]

def create_and_push(
        dev_type: DeviceType, cmd_id: hex, data: dict) -> None:
    global send_queue
    send_queue += [(dev_type, create_packet(cmd_id, data))]


def _identifier(hz_uart=2, hz_usb=2) -> None:

    def get_id_queue() -> AtomicList:
        global id_queue
        return id_queue

    def free_from_id_queue(dev_type: DeviceType) -> None:
        global id_queue
        if dev_type in id_queue:
            id_queue.remove(dev_type)

    def push_to_listen_queue(
            dev_type: DeviceType, device: Communicator) -> None:
        global listen_queue
        listen_queue += [(dev_type, device)]

    def _id_uart(hz):
        serial_experiments = []
        paths, prev_paths = [set()] * 2

        #cmd_id = config.SELFCHECK_CMD_ID
        #data = {'mode': 'ID', 'debug_int': 0}
        cmd_id = config.SELFCHECK_CMD_ID
        data = {'mode': 'ID', 'debug_int': 3}
        packet = create_packet(UART, cmd_id, data)
        while True:
            known_paths = {dev.get_port()
                           for dev in serial_devices.values()
                           if dev is not None
                           if not dev.is_vacuum()}
            id_queue = get_id_queue()
            if not any(map(is_uart, id_queue)):
                for dev in serial_experiments:
                    delist_device(dev)
                serial_experiments = []
                paths, prev_paths = [set()] * 2
                time.sleep(1/hz)
                continue
            # Look for devices.
            # `list_uart_device_paths` returns a list or [None]
            paths = set(UARTCommunicator.list_uart_device_paths()) - known_paths
            if None not in paths:
                for pth in paths - prev_paths:
                    try:
                        serial_experiments += [get_communicator(
                                UART, config, serial_dev_path=pth)]
                        logger.info(
                                f'Added {serial_experiments[-1]} '
                                'as identification candidate')
                    # Do not skip on error.
                    # Prevent `pth' from going into `prev_paths'.
                    except:
                        paths -= {pth}
                prev_paths = paths

            # Poll found devices.
            for dev in serial_experiments.copy():
                try:
                    _send_packet_dev(dev, packet)
                    _receive_uart_packet_dev(dev)
                    received_packet = _read_packet_dev(dev)
                    dev_type = received_packet['debug_int'] - 127
                    if dev_type in id_queue:
                        # Success.
                        serial_experiments.remove(dev)
                        logger.info(f'Identified {dname(dev_type)}, '
                                    f'pushing {dev} to listen queue.')
                        push_to_listen_queue(dev_type, dev)
                        logger.info(f'Freeing {dname(dev_type)} '
                                    'from identification queue.')
                        free_from_id_queue(dev_type)
                except:
                    serial_experiments.remove(dev)
                    logger.info(
                            f'Removed {dev} as identification candidate')
                    delist_device(dev)
            time.sleep(1/hz)

    # XXX: Add when ready
    def _id_usb(hz):
        while True:
            time.sleep(1/hz)

    id_uart_thread = threading.Thread(target=_id_uart,
                                      args=(hz_uart,))
    id_usb_thread = threading.Thread(target=_id_usb,
                                     args=(hz_usb,))
    id_uart_thread.start()
    id_usb_thread.start()

# Starts listening for new devices and updates `unified_state'. The
# frequency parameter only controls the rate of scanning `listen_queue' and
# fetching the state from `serial_devices' and not the frequency of the
# Communicator objects (in `serial_devices') themselves.
def _listener(hz_pull=4, hz_push=200):

    def get_listen_queue() -> AtomicList:
        global listen_queue
        return listen_queue

    def free_from_listen_queue(dev_type: DeviceType) -> None:
        global listen_queue
        listen_queue.filter_in_place(lambda el: el[0]!=dev_type)

    def push_to_id_queue(dev_type: DeviceType) -> None:
        global id_queue
        id_queue += [dev_type]

    def assign_device(dev_type: DeviceType, device: Communicator) -> None:
        serial_devices[dev_type] = device

    def deassign_device(dev_type: DeviceType) -> None:
        device = get_device(dev_type)
        assign_device(dev_type, None)
        delist_device(device)

    def _queue_puller(hz):
        while True:
            # Listen on and assign new devices
            listen_queue = get_listen_queue()
            for dev_type, dev in listen_queue:
                try:
                    dev.start_listening()
                    assign_device(dev_type, dev)
                    logger.info(f'{dname(dev_type)} listening as '
                                f'{dev}.')
                    print(f'=> {dname(dev_type)}: '
                          f'{GREEN if dev else RED}{dev}{RESET}')
                except:
                    logger.error(f'{dname(dev_type)} failed to listen as '
                                 f'{dev}.')
                    logger.info(f'Returning {dname(dev_type)} '
                                'to the identification queue.')
                    push_to_id_queue(dev_type)
                finally:
                    logger.info(f'Freeing {dname(dev_type)} from listen queue.')
                    free_from_listen_queue(dev_type)
            time.sleep(1/hz)

    def _state_pusher(hz):
        while True:
            # Get output of known devices.
            for dev_type, dev in serial_devices.items():
                if dev is not None:
                    if not dev.is_vacuum():
                        try:
                            packet = _read_packet(dev_type)
                            unified_state.update(_read_packet(dev_type))
                        except:
                            logger.error('Lost connection to '
                                         f'{dname(dev_type)}, deassigning '
                                         f'{get_device(dev_type)}.')
                            deassign_device(dev_type)
                            logger.info(f'Pushing {dname(dev_type)} '
                                        'to identification queue.')
                            push_to_id_queue(dev_type)
                            print(f'=> {dname(dev_type)}: '
                                  f'{RED}{get_device(dev_type)}{RESET}')
                else:
                    if dev_type not in id_queue and \
                            (dev_type, _m) not in listen_queue:
                        logger.info(f'Pushing {dname(dev_type)} '
                                    'to identification queue.')
                        push_to_id_queue(dev_type)
            time.sleep(1/hz)

    queue_puller_thread = threading.Thread(target=_queue_puller,
                                           args=(hz_pull,))
    state_pusher_thread = threading.Thread(target=_state_pusher,
                                           args=(hz_push,))
    queue_puller_thread.start()
    state_pusher_thread.start()

def _sender(hz=200):

    def takeone_from_send_queue() -> bytes:
        return send_queue.take_one()

    def taken_from_send_queue(
            dev_type: DeviceType, n: int = 1) -> List[bytes]:
        return send_queue.take_n(n)

    while True:
        head = takeone_from_send_queue()
        if head is not None:
            dev_type, packet = head
            try:
                _send_packet(dev_type, packet)
            except:
                pass
        time.sleep(1/hz)

# Convert octal to binary representation and run tests.
def startup_tests(
        testable: List[Tuple[Callable, DeviceType]], verb=0o10) -> None:
    # i.e. 0o16 => reversed('1110') will run test_board_{latency,pingpong,crc}
    bit_lst = reversed(str(bin(verb))[2:])  # Remove '0b'
    selected_tests = [test
                      for bit, test in zip(bit_lst, testable)
                      if bit == '1']
    return [action(get_device(dev_type))
            for action, dev_type in selected_tests]

def main(args) -> None:
    # Fork off our communication threads.
    hz_id = hz_pull = 4
    identifier_thread = threading.Thread(target=_identifier,
                                         args=(hz_id, hz_id))
    listener_thread = threading.Thread(target=_listener,
                                       args=(hz_pull,))
    sender_thread = threading.Thread(target=_sender)

    identifier_thread.daemon = True
    listener_thread.daemon = True
    sender_thread.daemon = True

    identifier_thread.start()
    listener_thread.start()
    sender_thread.start()

    # Run startup tests.
    if not args.skip_tests:
        testable: List[Tuple[Callable, DeviceType]] = [
                (communicator.test_board_typea, BRD),     # 0o1
                (communicator.test_board_crc, BRD),       # 0o2
                (communicator.test_board_pingpong, BRD),  # 0o4
                (communicator.test_board_latency, BRD),   # 0o10
                ]
        # [d]ev_[t]ype, [dev]ice, [req]uired
        dt_dev_reqset = {(dt, get_device(dt))
                         for _, dt in testable
                         if dt is not None}
        dt_reqlst, dev_reqlst = [*zip(*dt_dev_reqset)]  # Unzip into two lists
        if any(dev is None
               for dev in dev_reqlst):
            print('==> Not all devices required for testing attached, waiting.')
            for dt, dev in dt_dev_reqset:
                print(f'=> {dname(dt)}: {GREEN if dev else RED}{dev}{RESET}')
        # Wait for devices to be ready.
        while any(get_device(dt) is None
                  for dt in dt_reqlst):
            time.sleep(1/hz_id)
        print('==> All devices attached, continuing.\n')
        try:
            startup_tests(testable, int(args.test, 8))
        except Exception as e:
            logger.error('Startup tests failed, exiting...')
            raise e
    if args.test_only:
        exit(0)

    # Main loop
    j=0
    #cmd_id = config.SELFCHECK_CMD_ID
    cmd_id = config.ARM_CMD_ID
    while True:
        j += 1
        #data = {'mode': 'ID', 'debug_int': 0}
        data = {'floats':
                {
                    'float0': j,
                    'float1': j,
                    'float2': j,
                    'float3': j,
                    'float4': j,
                    'float5': j,
                    }
                }
        try:
            packet = create_packet(BRD, cmd_id, data)
            push_to_send_queue(BRD, packet)
            print(unified_state.deepcopy()['floats'])
        except:
            pass
        time.sleep(1)
        

if __name__ == '__main__':
    """ Items in parsed_args:
    parsed_args.verbosity: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
                                                      (default WARNING,
                                                       w/o arg INFO)
    parsed_args.test: 'octal'                         (d. '0o10')
    parsed_args.skip_tests: boolean
    parsed_args.test_only: boolean
    """
    # Remove first arg if called with python.
    _sl = slice(2, None) if 'python' in sys.argv[0] \
                         else slice(1, None)
    parsed_args = ap.parse_args(sys.argv[_sl])
    # Set up logging.
    set_up_loggers(loggers, parsed_args.verbosity)
    logger.debug(parsed_args)

    try:
        main(parsed_args)
    except KeyboardInterrupt:
        print()
        pass
