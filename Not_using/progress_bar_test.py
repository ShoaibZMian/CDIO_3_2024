import time
import random
from threading import Thread, Lock

def progress_bar(current, total, name, position, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>' 
    padding = int(bar_length - len(arrow)) * ' '

    with progress_lock:
        print(f'{LINE_UP*position}{name}: \t[{arrow}{padding}] {int(fraction*100)}%',end='\r')
        print(f'{LINE_DOWN*position}\r',end='\r')

s_print_lock = Lock()
def s_print(*a, **b):
    with s_print_lock:
        print(*a, **b)

def process_image():
    run_count = 0
    while run_count <= 100:
        time.sleep(0.25)
        if (random.random() >= 0.5):
            progress_bar(current=run_count, total=100, position=3, name="Processing image")
            run_count += 5

def calc_next_move():
    run_count = 0
    while run_count <= 100:
        time.sleep(0.25)
        if (random.random() >= 0.5):
            progress_bar(current=run_count, total=100, position=2, name="Calculating next move")
            run_count += 5

def send_to_robot():
    run_count = 0
    while run_count <= 100:
        time.sleep(0.25)
        if (random.random() >= 0.5):
            progress_bar(current=run_count, total=100, position=1, name="Sending to robot")
            run_count += 5


progress_lock = Lock()
LINE_UP = '\033[1A'
LINE_DOWN = '\033[1B'

print('\033[2K')
print('\033[2K')
print('\033[2K')
print('\033[2K')

t1 = Thread(target=process_image)
t2 = Thread(target=calc_next_move)
t3 = Thread(target=send_to_robot)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()