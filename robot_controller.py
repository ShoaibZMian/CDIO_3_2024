from abclasses import RobotControllerABC
import utils.status_printer as sp
import socket

class RobotController(RobotControllerABC):
    messages = ["Waiting for commands...",
                "Executing commands"]
    def __init__(self):
        sp.set_status(1, __class__.__name__, self.messages[0], 1)
        target_host = "172.20.10.4"
        target_port = 8080

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(1)
        self.client.connect((target_host, target_port))
        try:
                # receive the initial "ready" response
                response = self.client.recv(4096)
                if response.decode() == "ready":
                    print("Server is ready.")
                else:
                    print("Server is not ready.")

                # Set a longer timeout or remove it for waiting on the response
                self.client.settimeout(None)  # Wait indefinitely for the response

                # receive the result from the server
                while True:
                    try:
                        result = self.client.recv(4096)
                        if result:
                            print("Received response from server: ", result.decode())
                            break
                    except socket.timeout:
                        print("Still waiting for the server response...")
                        continue  # Keep waiting

        except socket.timeout:
            print("Connection timed out.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def send_move(self, command):
        sp.set_status(1, __class__.__name__, self.messages[1], 1)
        status_code = self.client.send(command.encode())
        sp.set_status(1, __class__.__name__, self.messages[0], 1)
        return status_code