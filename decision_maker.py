from time import sleep
from abclasses import DecisionMakerABC # type: ignore
import status_printer as sp # type: ignore

class DecisionMaker(DecisionMakerABC):
    def __init__(self, image_stack, condition):
        self.image_stack = image_stack
        self.condition = condition
        self.messages = ["Waiting for image...",
                         "Calculating next move for image "]
        self.current_image = -1
        sp.set_status(2, __class__.__name__, "Initialized")

    def start(self):
        sp.set_status(2, __class__.__name__, "Waiting for image...")
        while True:
            with self.condition:
                while len(self.image_stack) < 1:
                    self.condition.wait()
        
            self.current_image = self.image_stack.pop()
            sp.set_status(2, __class__.__name__, self.messages[1] + str(self.current_image))

    def start_demo(self):
        sp.set_status(2, __class__.__name__, "Waiting for image...")
        while True:
            with self.condition:
                while len(self.image_stack) < 1:
                    self.condition.wait()
        
            self.current_image = self.image_stack.pop()
            sp.set_status(2, __class__.__name__, self.messages[1] + str(self.current_image))
            sleep(5)


