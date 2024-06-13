from abclasses import ImageAnalyzerABC
import utils.status_printer as sp
from time import sleep
import logging
from image_data import ProcessedImageData
from threading import Condition
from segmentation import object_detection_opencv

class ImageAnalyzer(ImageAnalyzerABC):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.debug("testing logger from ImageAnalyzer")
    def __init__(self, shared_processed_image, condition):
        self.shared_processed_image: ProcessedImageData = shared_processed_image
        self.condition: Condition = condition
        self.messages = ["Analyzing image "]
        self.logger.debug("this is a debug log")
        sp.set_status(3, __class__.__name__, "Initialized")

    def start(self):
        print("starting")
        object_detection_opencv(self.shared_processed_image, self.condition)

    def start_demo(self):
        i = 1
        while True:
            sp.set_status(3, __class__.__name__, self.messages[0] + str(i))
            sleep(1)

            with self.condition:
                self.shared_processed_image.is_fresh = False
                self.condition.notify_all()
            i += 1

    def stop(self):
        pass