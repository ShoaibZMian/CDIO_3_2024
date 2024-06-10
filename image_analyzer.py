from abclasses import ImageAnalyzerABC # type: ignore
import status_printer as sp # type: ignore
from time import sleep
import logging

class ImageAnalyzer(ImageAnalyzerABC):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    def __init__(self, image_stack, condition):
        self.image_stack = image_stack
        self.condition = condition
        self.messages = ["Analyzing image "]
        sp.set_status(3, __class__.__name__, "Initialized")
        self.logger.debug("this is a debug log")

    def start(self):
        i = 1
        while True:
            sp.set_status(3, __class__.__name__, self.messages[0] + str(i))
            sleep(3)
            self.image_stack.append(i)

            if len(self.image_stack) == 1:
                with self.condition:
                    self.condition.notify_all()
            i += 1

    def start_demo(self):
        i = 1
        while True:
            sp.set_status(3, __class__.__name__, self.messages[0] + str(i))
            sleep(2)
            self.image_stack.append(i)

            if len(self.image_stack) == 1:
                with self.condition:
                    self.condition.notify_all()
            i += 1

    def stop(self):
        pass