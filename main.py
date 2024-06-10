import os
from robot_controller import RobotController # type: ignore
from image_analyzer import ImageAnalyzer # type: ignore
from decision_maker import DecisionMaker # type: ignore
from threading import Thread, Condition
from collections import deque
import keyboard
import status_printer as sp # type: ignore
from image_data import ImageData # type: ignore


# set quit hotkey    
keyboard.add_hotkey('q', lambda: os._exit(1))

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear') 
# hide cursor at end of line
print('\033[?25l', end="")

shared_image = ImageData()
condition = Condition()
ia = ImageAnalyzer(shared_image, condition)
dm = DecisionMaker(shared_image, condition)
rc = RobotController()

t_ia = Thread(target=ia.start_demo)
t_dm = Thread(target=dm.start_demo)
t_rc = Thread(target=rc.start_demo)

t_ia.start()
t_dm.start()
t_rc.start()

t_ia.join()
t_dm.join()
t_rc.join()

print("Tasks finished!")