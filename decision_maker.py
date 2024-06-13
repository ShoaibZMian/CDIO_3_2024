from time import sleep
from abclasses import DecisionMakerABC
import utils.status_printer as sp
from image_data import ProcessedImageData
from robot_controller import RobotController
import numpy as np

class DecisionMaker(DecisionMakerABC):
    def __init__(self, robot_controller, shared_processed_image, condition):
        self.robot_controller: RobotController = robot_controller
        self.shared_processed_image = shared_processed_image
        self.current_processed_image = ProcessedImageData()
        self.condition = condition
        self.target = self.closest_object
        self.messages = ["Waiting for image...",
                         "Calculating next move for image ",
                         "ERROR: "]
        sp.set_status(2, __class__.__name__, "Initialized")

    def closest_object(self):
        closest_distance = float('inf')
        closest_ball = None

        for green_object in self.current_processed_image.tracked_objects:
            if green_object.object_type == "green":
                for ball_object in self.current_processed_image.tracked_objects:
                    if ball_object.object_type == "ball":
                        distance = np.linalg.norm(np.array(green_object.centroid) - np.array(ball_object.centroid))
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_ball = ball_object
        return closest_ball

    def calculate_next_move(self) -> None:
        test = self.closest_object
        if self.closest_object != test:
            self.robot_controller.send_move(test.object_type)
                


    def start(self):
        sp.set_status(2, __class__.__name__, "Waiting for image...")
        while True:
            with self.condition:
                if not self.shared_processed_image.is_fresh:
                    sp.set_status(2, __class__.__name__, self.messages[0])
                    self.condition.wait()

                self.current_processed_image = self.shared_processed_image
                self.shared_processed_image.is_fresh = False

            sp.set_status(2, __class__.__name__, self.messages[1] + str(self.current_processed_image.id))

            next_move = self.calculate_next_move()
            status = self.robot_controller.send_move(next_move)

            match status:
                case _, 0: sp.set_status(3, __class__.__name__, self.messages[2] + "Command failed!")
                case 1: sp.set_status(3, __class__.__name__, "Command executed successfully!")