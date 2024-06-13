from abclasses import RobotControllerABC
import utils.status_printer as sp
class RobotController(RobotControllerABC):
    messages = ["Waiting for commands...",
                "Executing commands"]
    def __init__(self):
        sp.set_status(1, __class__.__name__, self.messages[0], 1)
    
    def send_move(self, command):
        pass