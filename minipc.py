#!/usr/bin/env python3
""" Extending this script:
    When making modifications, change:
        0) The configuration file (config.py)

        1) To add a new logger, modify:
        1.1) loggers  = [ (minipc.py:38)

        2) The identification scheme
        2.1) class Dev(IntEnum): (minipc.py:43)
        2.2) intenum_to_name = { (-:57)
        2.3) if parsed_args.plug_test: (-:389)
        2.4) serial_devices = { (-:69)

        3) The secion in __name__ == __main__
        3.1) serial_devices[ (-:395)

        4) argparse
        4.1) ap = argparse (minipc.py:77)
"""
import sys
import time
import logging
import argparse
from enum import IntEnum
from collections import namedtuple
from subprocess import run as sp_run

import config  # TODO: support multiple config files/config.Config1 etc

from Utils import color_logging as cl
from Utils.ansi import *
from Communication import communicator
from Communication.communicator import Communicator, UARTCommunicator, USBCommunicator
from Communication.packet import UARTPacket, USBPacket

# Add loggers here
loggers = [logger := logging.getLogger(__name__),
           c_logger := logging.getLogger('Communication.communicator')]


# UART > 0, USB <= 0
class Dev(IntEnum):
    UART = 1  # serialless uart, also acts to `type' others
    BRD = 2  # board
    SPM = 3  # spacemouse

    USB = 0  # serialless usb, also acts to `type' others
    ARM = -1  # arm


UART = Dev.UART
USB = Dev.USB
BRD = Dev.BRD
SPM = Dev.SPM
ARM = Dev.ARM

intenum_to_name = {
    UART: 'UART',
    USB: 'USB',
    BRD: 'BOARD',
    SPM: 'SPACEMOUSE',
    ARM: 'ARM',
}


def get_name(intenum):
    return intenum_to_name[intenum]


# Muting data in an (Int)Enum is not expected behavior, so:
serial_devices = {
    UART: UARTCommunicator(config, serial_dev_path=False, warn=False),
    USB: None,  # XXX: Replace when USBComm. is ready
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
# TODO: allow numeric
ap.add_argument('-v', '--verbosity',
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


def is_uart(dev):
    return dev >= UART


def is_usb(dev):
    return dev <= USB

#
# For unwrapped packets
#


def write_packet_uw(cmd_id, data, dev=UART):
    if is_uart(dev):
        uart = serial_devices[dev]
        unwrapped_packet = uart.create_packet(cmd_id, data)
    elif is_usb(dev):
        usb = serial_devices[dev]
        unwrapped_packet = None
    else:
        logger.error(f'Failed to encode: not a serial device: {dev}')
    return unwrapped_packet


def send_packet_uw(packet, dev=UART):
    if is_uart(dev):
        uart = serial_devices[dev]
        send_ret = uart.send_packet(packet)
    elif is_usb(dev):
        usb = serial_devices[dev]
        send_ret = None
    else:
        logger.error(f'Failed to send: not a serial device: {dev}')
    return send_ret


def write_send_packet_uw(cmd_id, data, dev=UART):
    unwrapped_packet = write_packet_uw(cmd_id, data, dev)
    send_ret = send_packet_uw(unwrapped_packet)
    return (unwrapped_packet, send_ret)
#
# For wrapped packets
#


def write_packet(cmd_id, data, dev=UART, dump=False):
    if is_uart(dev):
        uart = serial_devices[dev]
        packet = UARTPacket(cmd_id, data, uart)
    elif is_usb(dev):
        usb = serial_devices[dev]
        packet = None
    else:
        logger.error(f'Failed to encode Packet: not a serial device: {dev}')

    if packet is not None and dump:
        logging.debug(packet.dump_self(dump))

    return packet


def send_packet(packet, dev=UART, dump=False):
    if dump:
        logging.debug(packet.dump_self(dump))

    if is_uart(dev):
        uart = serial_devices[dev]
        send_ret = uart.send_packet(packet.contents)
    elif is_usb(dev):
        usb = serial_devices[dev]
        send_ret = None
    else:
        logger.error(f'Failed to send Packet: not a serial device: {dev}')

    return send_ret

# XXX: currently only latency, crc modified to work


def startup_tests(verb: oct = 0o10):
    # [(test, how the test should be run),]
    # tests should be an octal value:
    _tst_tup = namedtuple('TestTuple', 'action device')
    testable = [_tst_tup(communicator.test_board_latency, BRD),   # 0o10
                _tst_tup(communicator.test_board_pingpong, BRD),  # 0o4
                _tst_tup(communicator.test_board_crc, BRD),       # 0o2
                _tst_tup(communicator.test_board_typea, BRD)      # 0o1
                ]
    # Convert octal to binary representation and run tests.
    # i.e. oct(0o16) -> '001110' runs test_board_{latency,pingpong,crc}.
    selected_tests, selected_tests_return = [], []
    lst = str(format(verb, '#06b')[2:])
    for i in range(min(len(lst), len(testable))):
        if lst[i] == '1':
            selected_tests += [testable[i]]
    for tup in selected_tests:
        dev = serial_devices[tup.device]
        selected_tests_return += [tup.action(dev)]

    return selected_tests_return


def receive_packet(dev=UART):
    if is_uart(dev):
        uart = serial_devices[dev]
        if dev in (UART, BRD):
            return uart.get_current_stm32_state()
        else:
            return None
    elif is_usb(dev):
        return None
    else:
        logger.error(f'Failed to receive packet: not a serial device: {dev}')


def assign_device(dev_type, *args, **kwargs):
    if is_uart(dev_type):
        return UARTCommunicator(*args, **kwargs)
    elif is_usb(dev_type):
        # XXX: Replace when USBComm. is ready
        return UARTCommunicator(*args, **kwargs)
    else:
        logger.error(f'Failed to encode: not a serial device: {dev}')

# TODO: Add `udevadm` integration?


def perform_plug_test(
        *args: (Dev, Communicator)) -> [Communicator]:
    dev_lst = [tup[0] for tup in args]
    srl_lst = [tup[1] for tup in args]
    dev_srl_lst = [*args]
    print('==> Current port assignments:')
    for dev, srl in dev_srl_lst:
        print(f'=> {get_name(dev):<15} %s' %
              f'{RED}(no serial assigned){RESET}' if
              not srl else srl.serial_port)
    # If all serial ports are assigned
    if None not in [tup[1] for tup in args]:
        print('==> Nothing left to assign.')
        return srl_lst

    ret_list = []  # Of serial devices
    print(f'==> Please {YELLOW}_unplug_{RESET} all devices to be assigned.')
    input('=> (enter to continue) ')
    for dev, srl in dev_srl_lst:
        blacklisted_devices = [
            *UARTCommunicator.list_uart_device_paths(),
            # XXX: Replace when USBComm. is ready
            # *USBCommunicator.list_usb_device_paths())
        ]
        print(f'==> Blacklisted possible devices: {blacklisted_devices}')
        print(f'==> Please {GREEN}_plug_{RESET} the {get_name(dev)}.')
        input('=> (enter to continue) ')
        new_devices = [
            *UARTCommunicator.list_uart_device_paths(),
            # XXX: Replace when USBComm. is ready
            # *USBCommunicator.list_usb_device_paths())
        ]
        new_devices = [d for d in new_devices if d not in blacklisted_devices]
        if not new_devices:
            print(f'==> {YELLOW}No new devices found, '
                  f'assigning to serialless device.{RESET}')
            ret_list += [assign_device(dev, config, serial_dev_path=False)]
        else:
            if (_len := len(new_devices)) > 1:
                print(f'==> {YELLOW}More than one device found.{YELLOW}')
                print('==> It may be helpful to enter a number:')
                print('\n'.join([*map(lambda t: f'=> {t[0]+1}: {t[1]}',
                                      enumerate(new_devices))]))
                _nd_ind = input(f'=> (1 through {_len}) ')
                print(f'=> {GREEN}Using device: {new_devices[_nd_ind]}{RESET}')
                new_device = new_devices[_nd_ind]
            else:
                print(f'=> {GREEN}Using device: {new_devices[0]}{RESET}')
                new_device = new_devices[0]
            ret_list += [assign_device(dev, config, serial_dev_path=new_device)]
    print('\n==> New port assignments:')
    for dev, srl in zip(dev_lst, ret_list):
        print(f'=> {get_name(dev):<15} %s' %
              (f'{RED}(no port assigned){RESET}' if
               (b := not srl.serial_port) else srl.serial_port))
    print()

    return ret_list


def set_up_loggers(loggers):
    ch = logging.StreamHandler()
    ch.setFormatter(cl.ColorFormatter())
    for lgr in loggers:
        lgr.handlers.clear()
        lgr.addHandler(ch)
        lgr.setLevel(parsed_args.verbosity or 'DEBUG')


def set_up_devices(
        *args: (Dev, str), plug_test) -> [Communicator or None]:
    dev_lst = [tup[0] for tup in args]
    pth_lst = [tup[1] for tup in args]
    dev_pth_lst = [*args]

    ret_list = []  # Of serial devices or None
    for dev, pth in dev_pth_lst:
        if pth is not None:
            ret_list += [assign_device(
                dev, config, serial_dev_path=pth)]
        elif not plug_test:
            ret_list += [assign_device(dev, config)]
        else:
            ret_list += [None]

    return ret_list


def main(args):
    ########################
    # Run self check tests #
    ########################
    startup_tests(int(args.test, 8))
    if parsed_args.test_only:
        exit(0)

    ###################
    # Begin listening #
    ###################
    listening_devices = []
    for identifier, dev in serial_devices.items():
        if dev is not None and dev.serial_port is not None:
            dev.start_listening()
            print(f'{intenum_to_name[identifier]} listening on '
                  f'{dev.serial_port}')
            listening_devices += [dev]

    if not listening_devices:
        logger.error('No devices attatched: exiting...')
        exit(0)

    #####################
    #  Begin main loop  #
    #####################
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
    _sl = slice(2, None) if 'python' in sys.argv[0] else slice(1, None)
    """ Items in parsed_args:
    parsed_args.verbosity :: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
                                                        (d. WARNING,
                                                         w/o arg DEBUG)
    parsed_args.plug_test :: bool
    parsed_args.board_port :: '/path/to/some/port'      (d. None)
    parsed_args.spacemouse_port :: '/path/to/some/port' (d. None)
    parsed_args.arm_port :: '/path/to/some/port'        (d. None)
    parsed_args.test :: oct                             (default 0o10)
    parsed_args.skip_tests :: bool
    parsed_args.test_only :: bool
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
    brd_dev, spm_dev, arm_dev = \
        set_up_devices((BRD, parsed_args.board_port),
                       (SPM, parsed_args.spacemouse_port),
                       (ARM, parsed_args.arm_port),
                       plug_test=parsed_args.plug_test)
    #
    # Plug test
    #
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