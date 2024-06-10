from abclasses import ImageAnalyzerABC # type: ignore
import status_printer as sp # type: ignore
from time import sleep
import logging

class ImageAnalyzer(ImageAnalyzerABC):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    def __init__(self, shared_image, condition):
        self.shared_image = shared_image
        self.condition = condition
        self.messages = ["Analyzing image "]
        sp.set_status(3, __class__.__name__, "Initialized")
        self.logger.debug("this is a debug log")

    def start(self):
        pass

    def start_demo(self):
        i = 1
        self.shared_image.num = i
        while True:
            sp.set_status(3, __class__.__name__, self.messages[0] + str(i))
            sleep(1)

            with self.condition:
                self.shared_image.num = i
                self.condition.notify_all()
            i += 1

    def stop(self):
        pass