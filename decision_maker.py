from time import sleep
from abclasses import DecisionMakerABC # type: ignore
import status_printer as sp # type: ignore
from image_data import ImageData # type: ignore

class DecisionMaker(DecisionMakerABC):
    def __init__(self, shared_image, condition):
        self.shared_image = shared_image
        self.current_image = ImageData()
        self.condition = condition
        self.messages = ["Waiting for image...",
                         "Calculating next move for image "]
        sp.set_status(2, __class__.__name__, "Initialized")

    def start(self):
        pass

    def start_demo(self):
        sp.set_status(2, __class__.__name__, "Waiting for image...")
        while True:
            with self.condition:
                if self.shared_image.num == -1:
                    self.condition.wait()
    
                self.current_image.num = self.shared_image.num
                self.shared_image.num = -1

            sp.set_status(2, __class__.__name__, self.messages[1] + str(self.current_image.num))
            sleep(4)


