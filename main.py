import cv2
import yaml
import math
import socket
import logging
import os
import numpy as np
import time
from ultralytics import YOLO
from enum import Enum

class Phase(Enum):
    NOT_SET = 0
    BALL = 1,
    DELIVER = 2

class DeliverStep(Enum):
    NOT_SET = 0
    APPROACH_MIDPOINT = 1,
    APPROACH_GOAL = 2,
    AT_GOAL = 3

class CheckStep(Enum):
    NOT_SET = 0,
    FIRST = 1,
    SECOND = 2


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')


class DetectedObject:
    def __init__(self, name, x1, y1, x2, y2, confidence, midpoint):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.midpoint = midpoint

class data_model:
    def __init__(self):
        self.balls = []
        self.corners = []
        self.front:DetectedObject
        self.back:DetectedObject
        self.obstacle:DetectedObject
        self.egg:DetectedObject
        self.small_goal:DetectedObject

current_data_model = data_model()
phase = Phase.BALL
deliver_step = DeliverStep.NOT_SET
number_of_balls = -1
check_step = CheckStep.FIRST

model_path = "/Users/matt/CDIO_3_2024/v14/best.pt"
data_yaml_path = "/Users/matt/CDIO_3_2024/v14/data.yaml"
video_path = 0
#conf_thresholds = {'white-golf-ball': 0.5,'robot-front': 0.25,'robot-back': 0.25,'egg': 0.5,"corner1": 0.60}
conf_thresholds = {'corner1': 0.5, 'egg': 0.5, 'obstacle': 0.5, 'orange-golf-ball': 0.5, 'robot-back': 0.1, 'robot-front': 0.1, 'small-goal': 0.5, 'white-golf-ball': 0.35}


def calculate_midpoint(x1,y1,x2,y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']           

def draw_boxes(frame, results, class_names):
    for result in results:
        for obj in result.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            class_index = int(obj.cls)
            name = class_names[class_index]
            confidence = obj.conf
            label = f"{name} {confidence}"
            if name != "robot":
                if name in conf_thresholds and confidence >= conf_thresholds[name]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def update_list(results, class_names, frame):
    current_data_model.balls.clear()
    current_data_model.corners.clear()

    for result in results:
        for obj in result.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            class_index = int(obj.cls)
            name = class_names[class_index]
            confidence = obj.conf
            midpoint = calculate_midpoint(x1,y1,x2,y2)
            new_object = DetectedObject(name, x1, y1, x2, y2, confidence, midpoint)

            if name in conf_thresholds and confidence >= conf_thresholds[name]:
                match new_object.name:
                    case "corner1":
                        current_data_model.corners.append(new_object)
                    case "egg":
                        current_data_model.egg = new_object
                    case "obstacle":
                        current_data_model.obstacle = new_object
                    case "white-golf-ball" | "orange-golf-ball":
                        current_data_model.balls.append(new_object)
                    case "robot-back":
                        current_data_model.back = new_object
                    case "robot-front":
                        current_data_model.front = new_object
                    case "small-goal":
                        current_data_model.small_goal = new_object
        
def calculate_angle(midpoint):
    if not hasattr(current_data_model, "front"):
        print("Error: 'robot-front' not found in data model")
        return None
    
    if not hasattr(current_data_model, "back"):
        print("Error: 'robot-back' not found in data model")
        return None
    
    x0, y0 = current_data_model.back.midpoint
    x1, y1 = current_data_model.front.midpoint
    x2, y2 = midpoint

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
    
    angle_radians = np.arccos(cos_theta)
    
    if cross_product > 0:
        angle_radians = -angle_radians
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    # Check if angle is within desired range
    return int(angle_degrees)

def calculate_distance(midpoint, end_of_robot): 
    if current_data_model.front is None:
        print("Error: 'robot-front' not found in shared_list_copy")
        return -1
    if end_of_robot == "front":
        current_x, current_y = current_data_model.front.midpoint
    elif end_of_robot == "back":
        current_x, current_y = current_data_model.front.midpoint

    target_x, target_y = midpoint
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    distance = math.sqrt(delta_x**2 + delta_y**2)
    return int(distance)

def start_client():
    #target_host = "192.168.18.18"
    target_host = "172.20.10.4"
    #target_host = "192.168.105.18"
    #target_host = "localhost"
    target_port = 8080
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect((target_host, target_port))
        client.settimeout(5)
        response = client.recv(4096)
        if response.decode() == "ready":
            logging.debug("Server is ready.")
        else:
            logging.debug("Server is not ready.")
        client.settimeout(None)
    except socket.timeout:
        logging.error("Connection timed out.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return client

def send_command(command):
    client_socket = start_client()
    if client_socket is None:
        logging.error(f"An error occurred while starting server")
        return
    client_socket.send(command.encode())
    response = client_socket.recv(4096)
    #logging.debug("Received response from server: ", response.decode())
    return response.decode()


def get_closest_ball():
    #logging.debug("Runnign Closest ball function")
    closest_distance = float('inf')
    closest_ball = None

    if not hasattr(current_data_model, "front"):
        wiggle()
    for ball in current_data_model.balls:
        distance = np.linalg.norm(np.array(current_data_model.front.midpoint) - np.array(ball.midpoint))
        if distance < closest_distance:
            closest_distance = distance
            closest_ball = ball
    return closest_ball

def move_to_ball(closest_ball):
    global phase 
    if closest_ball is None:
        logging.debug("Found no close ball")
        check_ball_capture()
        wiggle()
        if get_closest_ball() == None:
            phase = Phase.DELIVER
        return

    angle = calculate_angle(closest_ball.midpoint)
    distance = calculate_distance(closest_ball.midpoint, "front")

    
    global number_of_balls
    global check_step
    if number_of_balls > len(current_data_model.balls):
        check_ball_capture()
        wiggle()
        if check_step == CheckStep.FIRST:
            logging.debug("first checkstep")
            check_step = CheckStep.SECOND
            return
        if check_step == CheckStep.SECOND:
            logging.debug("second checkstep")
            number_of_balls -= 1
            check_step = CheckStep.FIRST
            if number_of_balls % 5 == 0:
                phase = Phase.DELIVER
            return
    
    if angle is not None and (-10 > angle or angle > 10):
        command = f"turn{angle}"
        logging.debug(f"send command: {command}")
        send_command(command)
        return
    if calculate_distance(closest_ball.midpoint, "front") > 10:
        command = f"drive{distance}"
        logging.debug(f"send command: {command}")
        send_command(command)
        return
        
def wiggle() -> None:
    logging.debug("robot is wiggling")
    send_command("turn-10")
    send_command("turn+20")
    send_command("turn-10")

def check_ball_capture() -> None:
    logging.debug("performing ball check movement")
    send_command("drive+10")
    send_command("drive-30")

def calculate_mask(frame, obj):
    cropped = frame[obj.y1:obj.y2, obj.x1:obj.x2] 
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV) 
    if obj.name == "robot-back": 
        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([160, 255, 255])
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        contours_purple, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_purple
    else:
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours_green

def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    else:
        return None

def is_data_model_ready():
    if current_data_model is not None and current_data_model.balls and current_data_model.corners:
        return True
    else:
        return False
 
def on_point(target):
    distance = calculate_distance(target, "back")
    logging.debug(f"distance to target: {distance}")
    return distance <= 120

def deliver_to_goal(client_socket):
    global deliver_step
    if deliver_step == DeliverStep.NOT_SET:
        deliver_step = DeliverStep.APPROACH_MIDPOINT

    step1_x = current_data_model.small_goal.midpoint[0] - 300
    midpoint = (step1_x, current_data_model.small_goal.midpoint[1])
    logging.debug(f"current deliverstep: {deliver_step.name}")
    match deliver_step:
        
        case DeliverStep.NOT_SET:
            logging.debug("error: deliverstep is not set")
        case DeliverStep.APPROACH_MIDPOINT:
            if  not on_point(midpoint):
                angle = calculate_angle(midpoint)
                distance = calculate_distance(midpoint, "front")
                logging.debug(f"angle: {angle} distance: {distance}")
                if angle is not None and (-10 > angle or angle > 10):
                    command = f"turn{angle}"
                else:
                    command = f"drive{distance}"
                logging.debug(f"send command: {command}")
                send_command(command)
            else:
                deliver_step = DeliverStep.APPROACH_GOAL
        case DeliverStep.APPROACH_GOAL:
            if not on_point((current_data_model.small_goal.midpoint[0] - 100, current_data_model.small_goal.midpoint[1])):
                angle_to_goal = calculate_angle(current_data_model.small_goal.midpoint)
            if angle_to_goal is not None and (-10 > angle_to_goal or angle_to_goal > 10):
                command = f"turn{angle_to_goal}"
            else:
                distance = calculate_distance(current_data_model.small_goal.midpoint, "front")
                command = f"drive{distance}"
                deliver_step = DeliverStep.AT_GOAL
            send_command(command)
            logging.debug(f"send command: {command}")
        case DeliverStep.AT_GOAL:
            angle_to_goal = calculate_angle(current_data_model.small_goal.midpoint)
            if angle_to_goal is not None and (-5 > angle_to_goal or angle_to_goal > 5):
                send_command(f"turn{angle_to_goal}")
            else:
                send_command("off0")
                logging.debug(f"send command: off0")
                time.sleep(5)
                send_command("on0")
                logging.debug(f"send command: on0")
                send_command("drive-30")
                logging.debug(f"send command: drive-30")
                send_command("turn180")
                logging.debug(f"send command: turn180")
            
                deliver_step = DeliverStep.NOT_SET
                global phase
                phase = Phase.BALL
        case _: 
            logging.debug("error: deliverstep is not set")
        

def controller(is_server_online, client_socket):
    global phase
    model = YOLO(model_path)
    class_names = load_class_names(data_yaml_path)
    cap = cv2.VideoCapture(video_path)

    time.sleep(2)

    if not cap.isOpened():
            logging.error("Error opening video stream or file")
            return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error("No frame available")
            break

        logging.debug(f"current phase: {phase.name}")
        results = model(frame, verbose=False, conf=0.40)
        update_list(results, class_names, frame)
        global number_of_balls
        if number_of_balls == -1:
            number_of_balls = len(current_data_model.balls)

        draw_boxes(frame, results, class_names)

        if len(current_data_model.balls) == 0:
            phase = Phase.DELIVER

        closest_ball = get_closest_ball()
        #if close_ball:
        #    cv2.line(frame, close_ball.midpoint, current_data_model.front.midpoint, (0, 0, 255), 2)
        if hasattr(current_data_model, "front") and hasattr(current_data_model, "back"):
            cv2.line(frame, current_data_model.front.midpoint, current_data_model.back.midpoint , (0, 0, 255), 2)
        if hasattr(current_data_model, "small_goal") and hasattr(current_data_model, "back"):
            cv2.line(frame, current_data_model.small_goal.midpoint, (current_data_model.back.midpoint[0] - 100, current_data_model.small_goal.midpoint[1]) , (0, 0, 255), 2)

        cv2.imshow('Frame', frame)
        logging.debug(closest_ball)

        if not is_server_online:
            client_socket = start_client()
            if client_socket:
                is_server_online = True
                logging.debug("Started client")
        if client_socket:
            if phase == Phase.BALL:
                move_to_ball(closest_ball)
            else:
                deliver_to_goal(client_socket)
            client_socket = False
            is_server_online = False

        #os.system("cls" if os.name == "nt" else "clear")
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    is_server_online = False
    client_socket = None
    controller(is_server_online, client_socket)
