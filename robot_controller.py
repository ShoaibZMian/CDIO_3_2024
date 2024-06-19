from abclasses import RobotControllerABC # type: ignore
import status_printer as sp # type: ignore
class RobotController(RobotControllerABC):
    messages = ["Waiting for commands...",
                "Executing commands"]
    def __init__(self):
        sp.set_status(1, __class__.__name__, self.messages[0], 1)

    def start(self):
        pass

    def start_demo(self):
        pass