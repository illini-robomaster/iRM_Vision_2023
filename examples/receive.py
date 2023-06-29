import time
import config
from Communication.communicator import UARTCommunicator


def main():
    communicator = UARTCommunicator(config)
    if communicator.is_valid():
        print("OPENED SERIAL DEVICE AT: " + communicator.serial_port.name)
    else:
        print("SERIAL DEVICE IS NOT AVAILABLE!!!")
    communicator.start_listening()
    while True:
        print(communicator.stm32_state_dict['cur_yaw'])
        time.sleep(0.2)

if __name__ == "__main__":
    main()
