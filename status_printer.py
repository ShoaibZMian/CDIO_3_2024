import threading

print_lock = threading.Lock()
LINE_UP = "\033[1A"
LINE_DOWN = "\033[1B"
tab = "\t"

def set_status(position, name, message="", tabs=2):
    with print_lock:
        print(f'{LINE_UP*position}{name}: {tab*tabs}{message}',end='\033[0K\r')
        print(f'{LINE_DOWN*position}\r',end='\r')
