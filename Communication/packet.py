#
#!/usr/bin/env python3
#
# Should be imported from package root
#
import config
from .communicator import UARTCommunicator

# UARTCommunicator packet, for easier inspection
class UARTPacket():
    def __init__(self, cmd_id, data, uart):
        self.cmd_id = cmd_id
        self.data = data
        self.uart = uart

        self.contents = self.uart.create_packet(cmd_id, data)
        self.dumpable = [self.uart,     # 0o10
                         self.contents, # 0o4
                         self.data,     # 0o2
                         self.cmd_id]   # 0o1
    #
    # Pass an octal value i.e. 0o14 = 0o10 + 0o4 = 1100
    # 1 -> output uart
    # 1 -> output byte contents
    # 0 -> don't output data
    # 0 -> don't output cmd_id
    #
    def dump_self(self, verb:oct):
        if verb is None:
            return None
        lst = str(format(verb, '#06b')[2:])
        ret = []
        for i in range(min(len(lst), len(self.dumpable))):
            if lst[i] == '1':
                ret += [self.dumpable[i]]
        return ret

class USBPacket:
    def __init__(self):
        pass


class Packet:
    pass
