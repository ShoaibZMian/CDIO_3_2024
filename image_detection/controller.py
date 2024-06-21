import threading
import time
import cv2
import yaml
import math
import socket
import numpy as np
from queue import Queue
from ultralytics import YOLO
from robot_controls.robot_client import robot_client_thread

shared_list = []
list_lock = threading.Lock()
robot_ready = threading.Condition()
frame_queue = Queue()

model_path = "/Users/matt/CDIO_3_2024/best.v12/best (1).pt"
data_yaml_path = "/Users/matt/CDIO_3_2024/best.v12/data.yaml"
video_path = 0
conf_thresholds = {'white-golf-ball': 0.4,'robot-front': 0.25,'robot-back': 0.25,'egg': 0.70,"corner1": 0.60}

robot_moving = False
is_server_online = False
client_socket = None

class DetectedObject:
    def __init__(self, name, x1, y1, x2, y2, confidence):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.midpoint = calculate_midpoint(x1, y1, x2, y2)

def calculate_midpoint(x1,y1,x2,y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']           

def draw_boxes(detected_objects, frame):
    for obj in detected_objects:
        if obj.name != "robot":
            x1, y1, x2, y2 = obj.x1, obj.y1, obj.x2, obj.y2
            label = f"{obj.name} {float(obj.confidence):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, obj.midpoint, radius=5, color=(0, 0, 255), thickness=-1)

def update_list(detected_objects, results, class_names, threshold=10):
    detected_objects.clear()
    
    for result in results:
        # Iterate over detected objects
        for detected in result.boxes:
            x1, y1, x2, y2 = map(int, detected.xyxy[0])
            class_index = int(detected.cls)
            name = class_names[class_index]
            confidence = detected.conf
            new_object = DetectedObject(name, x1, y1, x2, y2, confidence)
            detected_objects.append(new_object)

def calculate_angle(close_ball, shared_list_copy):
    # Find robot-front object
    robot_front_obj = next((obj for obj in shared_list_copy if obj.name == "robot-front"), None)
    if robot_front_obj is None:
        print("Error: 'robot-front' not found in shared_list_copy")
        return None
    
    # Find robot-back object
    robot_back_obj = next((obj for obj in shared_list_copy if obj.name == "robot-back"), None)
    if robot_back_obj is None:
        print("Error: 'robot-back' not found in shared_list_copy")
        return None
    
    x0, y0 = robot_back_obj.midpoint
    x1, y1 = robot_front_obj.midpoint
    x2, y2 = close_ball.midpoint

    # Create vectors from the coordinates
    vector1 = np.array([x1 - x0, y1 - y0])
    vector2 = np.array([x2 - x0, y2 - y0])
    
    # Calculate dot product and cross product
    dot_product = np.dot(vector1, vector2)
    cross_product = np.cross(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate cosine of the angle
    cos_theta = dot_product / (magnitude1 * magnitude2)
    
    # Clip cos_theta to handle floating point precision issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate angle in radians
    angle_radians = np.arccos(cos_theta)
    
    # Determine the sign (clockwise or counterclockwise) using the cross product
    # Compare vectors to determine the direction of rotation
    if cross_product > 0:
        angle_radians = -angle_radians
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    # Check if angle is within desired range
    return int(angle_degrees)

def calculate_distance(close_ball, shared_list_copy):
    robot_front_obj = None
    for obj in shared_list_copy:
        if obj.name == "robot-front":
            robot_front_obj = obj
            break
    
    if robot_front_obj is None:
        print("Error: 'robot-front' not found in shared_list_copy")
        return None, None
    
    current_x, current_y = robot_front_obj.midpoint
    target_x, target_y = close_ball.midpoint
    
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    distance = math.sqrt(delta_x**2 + delta_y**2)
    
    return int(distance)

def start_client():
    target_host = "172.20.10.4"
    #target_host = "localhost"
    target_port = 8080
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect((target_host, target_port))
        client.settimeout(5)
        response = client.recv(4096)
        if response.decode() == "ready":
            print("Server is ready.")
        else:
            print("Server is not ready.")
        client.settimeout(None)
    except socket.timeout:
        print("Connection timed out.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return client

def send_command(client_socket, command):
    try:
        client_socket.send(command.encode())
        response = client_socket.recv(4096)
        print(f"Server response: {response.decode()}")
        return response.decode()
    except Exception as e:
        print(f"An error occurred while sending command: {e}")

def closest_ball(shared_list_copy):
    closest_distance = float('inf')
    closest_ball = None

    for front in shared_list_copy:
        if front.name == "robot-front":
            for ball in shared_list_copy:
                if ball.name == "white-golf-ball":
                    distance = np.linalg.norm(np.array(front.midpoint) - np.array(ball.midpoint))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_ball = ball

    return closest_ball

def move_to_ball(shared_list_copy, client_socket, list_lock):
    global robot_moving
    robot_moving = True
    with list_lock:
        shared_list_copy = shared_list_copy.copy()
    close_ball = closest_ball(shared_list_copy)
    print(f"closest ball:{close_ball}")
    if close_ball is not None:
        angle = calculate_angle(close_ball, shared_list_copy)
        distance = calculate_distance(close_ball, shared_list_copy)
        if angle is not None and (-10 > angle or angle > 10):
            command = f"turn{angle}"
        else:
            command = f"drive{distance}"

        send_command(client_socket, command)
        #time.sleep(1)
    robot_moving = False

def controller():
    model = YOLO(model_path)
    class_names = load_class_names(data_yaml_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
            print("Error opening video stream or file")
            return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No frame available")
            break

        results = model(frame, verbose=False)
        draw_boxes(shared_list, frame)
        update_list(shared_list, results, class_names)

        if shared_list:
            if not is_server_online:
                client_socket = start_client()
                if client_socket:
                    is_server_online = True
                    print("Started client")
            if client_socket:
                move_to_ball(shared_list, client_socket, list_lock)
                client_socket = False
                is_server_online = False


        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    controller()
