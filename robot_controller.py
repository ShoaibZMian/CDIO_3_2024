from abclasses import RobotControllerABC
import utils.status_printer as sp
import socket

class RobotController(RobotControllerABC):
    messages = ["Waiting for commands...",
                "Executing commands"]
    def __init__(self):
        sp.set_status(1, __class__.__name__, "Initializing...", 1)
        target_host = "localhost"
        target_port = 8080

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(5)
        self.client.connect((target_host, target_port))
        try:
                # receive the initial "ready" response
                self.client.settimeout(5)
                response = self.client.recv(4096)
                if response.decode() == "ready":
                    print("Server is ready.")
                else:
                    print("Server is not ready.")

        except socket.timeout:
            print("Connection timed out.")
        except Exception as e:
            print(f"An error occurred: {e}")

        sp.set_status(1, __class__.__name__, self.messages[0], 1)
    
    def send_move(self, command):
        sp.set_status(1, __class__.__name__, self.messages[1], 1)
        status_code = self.client.send(command.encode())
        sp.set_status(1, __class__.__name__, self.messages[0], 1)
        return status_code