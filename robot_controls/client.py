import socket
import sys
import os
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_detection.yolov8Shweb import yolo_init
from image_detection.itemManager import get_all_items, reset
from robot_controls.robotManager import robot_process_items


yolo_model_path = "C:/Users/Mian/Desktop/bestv12/best.pt"
yolo_data_yaml_path = "C:/Users/Mian/Desktop/bestv12/train_folder/cdio3.v12i.yolov8/data.yaml"
yolo_video_path = "C:/Users/Mian/Downloads/filmm.mov"

# Define confidence thresholds for specific items
yolo_conf_thresholds = {
    'white-golf-ball': 0.4,  
    'robot-front': 0.25,     
    'robot-back': 0.25,      
    'egg': 0.70, 
}

def send_command(command):
    target_host = "172.20.10.4"
    target_port = 8080

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(1)
    client.connect((target_host, target_port))
    try:
            # receive the initial "ready" response
            response = client.recv(4096)
            if response.decode() == "ready":
                print("Server is ready.")
            else:
                print("Server is not ready.")

            # send the command
            client.send(command.encode())
            print(f"Sent command: {command}")

            # Set a longer timeout or remove it for waiting on the response
            client.settimeout(None)  # Wait indefinitely for the response

            # receive the result from the server
            while True:
                try:
                    result = client.recv(4096)
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
    finally:
        client.close()


def start_yolo():
    yolo_init(yolo_model_path, yolo_data_yaml_path, yolo_video_path, yolo_conf_thresholds)

def start_robot_process_items():
    items = get_all_items()
    print("\n".join([f"{time.time()}: {items}"]))
    reset()

    # check that items has what is needed before calling the robot
    required_items = ['robot-front', 'robot-back', 'white-golf-ball'] 
    if not all(item in items for item in required_items): # type: ignore
        print("Required items not found in 'items'. Function will exit.")
        return

    robot_process_items(items)

def periodic_task(interval, function, *args, **kwargs):
    """
    Runs a function periodically with the given interval.
    """
    while True:
        function(*args, **kwargs)
        time.sleep(interval)

if __name__ == "__main__":

    # Start the periodic task in a new thread
    interval = 5  # 5 seconds
    periodic_thread = threading.Thread(target=periodic_task, args=(interval, start_robot_process_items))
    periodic_thread.daemon = True
    periodic_thread.start()

    start_yolo()



    # while True:
    #     cmd = input(
    #         "Enter a command (e.g., 'forward10', 'backward20', or 'quit' to exit): "
    #     )
    #     if cmd.lower() == "quit":
    #         break
    #     send_command(cmd)