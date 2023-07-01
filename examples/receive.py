import time
import config
from Communication.communicator import UARTCommunicator


def main():
    communicator = UARTCommunicator(config)
    if communicator.is_valid():
        print('OPENED SERIAL DEVICE AT: ' + communicator.serial_port.name)
    else:
        print('SERIAL DEVICE IS NOT AVAILABLE!!!')
    communicator.start_listening()
    while True:
        print('my color: ', communicator.stm32_state_dict['my_color'])
        print('cur yaw: ', communicator.stm32_state_dict['cur_yaw'])
        print('cur pitch: ', communicator.stm32_state_dict['cur_pitch'])
        print('cur roll: ', communicator.stm32_state_dict['cur_roll'])
        print('timestamp: ', communicator.stm32_state_dict['timestamp'])
        print('')
        time.sleep(0.1)


if __name__ == "__main__":
    main()
