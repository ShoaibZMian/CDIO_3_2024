import os
from robot_controller import RobotController
from image_analyzer import ImageAnalyzer
from decision_maker import DecisionMaker
from threading import Thread, Condition
from collections import deque
#import keyboard
import utils.status_printer as sp
from image_data import ProcessedImageData
import cv2
import time


# set quit hotkey    
#keyboard.add_hotkey('q', lambda: os._exit(1))

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

time.sleep(5)
# clear terminal
os.system('cls' if os.name == 'nt' else 'clear') 
# hide cursor at end of line
print('\033[?25l', end="")
while True:
    if shared_image.frame is not None:
        cv2.imshow('Tracked Objects', shared_image.frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

t_ia.join()
t_dm.join()

cv2.destroyAllWindows()
print("Tasks finished!")