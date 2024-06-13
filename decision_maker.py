from time import sleep
from abclasses import DecisionMakerABC
import utils.status_printer as sp
from image_data import ProcessedImageData
from robot_controller import RobotController

class DecisionMaker(DecisionMakerABC):
    def __init__(self, robot_controller, shared_processed_image, condition):
        self.robot_controller: RobotController = robot_controller
        self.shared_processed_image = shared_processed_image
        self.current_processed_image = ProcessedImageData()
        self.condition = condition
        self.messages = ["Waiting for image...",
                         "Calculating next move for image ",
                         "ERROR: "]
        sp.set_status(2, __class__.__name__, "Initialized")

    def start_demo(self):
        sp.set_status(2, __class__.__name__, "Waiting for image...")
        while True:
            with self.condition:
                if not self.shared_processed_image.is_fresh:
                    self.condition.wait()
    
                self.current_image = self.shared_processed_image
                self.shared_processed_image.is_fresh = False

            sp.set_status(2, __class__.__name__, self.messages[1] + str(self.current_processed_image.id))
            sleep(4)

    def calculate_next_move(self) -> None:
        pass

    def start(self):
        sp.set_status(2, __class__.__name__, "Waiting for image...")
        while True:
            with self.condition:
                if not self.shared_processed_image.is_fresh:
                    self.condition.wait()
    
                self.current_processed_image = self.shared_processed_image
                self.shared_processed_image.is_fresh = False

            sp.set_status(2, __class__.__name__, self.messages[1] + str(self.current_image.id))

            next_move = self.calculate_next_move()
            status = self.robot_controller.send_move(next_move)

            match status:
                case _, 0: sp.set_status(3, __class__.__name__, self.messages[2] + "Command failed!")
                case 1: sp.set_status(3, __class__.__name__, "Command executed successfully!")