import time
import random
from threading import Thread, Lock

def set_status(position, name, message=""):
    with print_lock:
        print(f'{LINE_UP*position}{name}: \t{message}',end='\033[0K\r')
        print(f'{LINE_DOWN*position}\r',end='\r')

s_print_lock = Lock()
def s_print(*a, **b):
    with s_print_lock:
        print(*a, **b)

def process_image():
    set_status(3, "Image processing", "Initializing...")
    run_count = 0
    while run_count < 19:
        status_sim(3, "Image processing", "Waiting for new image...", "Processing...")
        time.sleep(1)
        run_count += 1

def calc_next_move():
    set_status(2, "Move calculator", "Initializing...")
    run_count = 0
    while run_count < 19:
        status_sim(2, "Move calculator", "Waiting for processed image...", "Calculating next move...")
        time.sleep(1)
        run_count += 1

def send_to_robot():
    set_status(1, "Robot control:", "\tInitializing...")
    run_count = 0
    while run_count < 19:
        status_sim(1, "Robot control", "\tSending commands...", "\tWaiting for response...")
        time.sleep(1)
        run_count += 1

def status_sim(position, name, message1, message2):
        if (random.random() >= 0.5):
            set_status(position=position, name=name, message=message1)
        else:
            set_status(position=position, name=name, message=message2)

print_lock = Lock()
LINE_UP = '\033[1A'
LINE_DOWN = '\033[1B'

# clear terminal
print('\033[1J',end='\r\n')

t1 = Thread(target=process_image)
t2 = Thread(target=calc_next_move)
t3 = Thread(target=send_to_robot)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()

print("Tasks finished!")