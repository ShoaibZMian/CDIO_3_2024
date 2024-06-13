import os
from robot_controller import RobotController
from image_analyzer import ImageAnalyzer
from decision_maker import DecisionMaker
from threading import Thread, Condition
from collections import deque
import keyboard
import utils.status_printer as sp
from image_data import ProcessedImageData


# set quit hotkey    
keyboard.add_hotkey('q', lambda: os._exit(1))

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear') 
# hide cursor at end of line
print('\033[?25l', end="")

shared_image = ProcessedImageData()
condition = Condition()
ia = ImageAnalyzer(shared_image, condition)
rc = RobotController()
dm = DecisionMaker(rc, shared_image, condition)

t_ia = Thread(target=ia.start)
t_dm = Thread(target=dm.start)

t_ia.start()
t_dm.start()

t_ia.join()
t_dm.join()

print("Tasks finished!")