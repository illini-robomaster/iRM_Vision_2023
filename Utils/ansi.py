#!/usr/bin/env python3
# ANSI escapes
BLACK = '\033[30m'
DARKGRAY = DARKGREY = '\033[90m'

DARKRED = '\033[31m'
RED = '\033[91m'

DARKGREEN = '\033[32m'
GREEN = '\033[92m'

DARKYELLOW = '\033[33m'
YELLOW = '\033[93m'

DARKBLUE = '\033[34m'
BLUE = '\033[94m'

DARKMAGENTA = '\033[35m'
MAGENTA = '\033[95m'

DARKCYAN = '\033[36m'
CYAN = '\033[96m'

LIGHTGRAY = LIGHTGREY = '\033[37m'
WHITE = '\033[97m'

UNDERLINE = '\033[4m'
NOUNDERLINE = '\033[24m'
BOLD = '\033[1m'
NOBOLD = '\033[21m'

RESET = '\033[39m\033[49m'
CLEAR = '\033[2K'


if __name__ == '__main__':
    print(BLACK + f'{BLACK=}' + RESET, end='\t')
    print(DARKGRAY + f'{DARKGRAY=}' + RESET, end='\t')
    print(DARKGREY + f'{DARKGREY=}' + RESET)
    print(DARKRED + f'{DARKRED=}' + RESET, end='\t')
    print(RED + f'{RED=}' + RESET)
    print(DARKGREEN + f'{DARKGREEN=}' + RESET, end='\t')
    print(GREEN + f'{GREEN=}' + RESET)
    print(DARKYELLOW + f'{DARKYELLOW=}' + RESET, end='\t')
    print(YELLOW + f'{YELLOW=}' + RESET)
    print(DARKBLUE + f'{DARKBLUE=}' + RESET, end='\t')
    print(BLUE + f'{BLUE=}' + RESET)
    print(DARKMAGENTA + f'{DARKMAGENTA=}' + RESET, end='\t')
    print(MAGENTA + f'{MAGENTA=}' + RESET)
    print(DARKCYAN + f'{DARKCYAN=}' + RESET, end='\t')
    print(CYAN + f'{CYAN=}' + RESET)
    print(LIGHTGRAY + f'{LIGHTGRAY=}' + RESET, end='\t')
    print(LIGHTGREY + f'{LIGHTGREY=}' + RESET, end='\t')
    print(WHITE + f'{WHITE=}' + RESET)

    print(UNDERLINE + f'{UNDERLINE=}' + NOUNDERLINE + '\t' + f'{NOUNDERLINE=}')
    print(BOLD + f'{BOLD=}' + NOBOLD + '\t\t' + f'{NOBOLD=}')

    print(f'{RESET=}')
    print(f'{CLEAR=}')
