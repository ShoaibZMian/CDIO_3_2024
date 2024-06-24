import cv2
import yaml
import math
import socket
import logging
import numpy as np
import time
from ultralytics import YOLO

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

model_path = "best14/best.pt"
data_yaml_path = "best14/data.yaml"
video_path = 1
conf_thresholds = {'white-golf-ball': 0.4,'robot-front': 0.25,'robot-back': 0.25,'egg': 0.70,"corner1": 0.60}


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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # for obj in current_data_model.balls + current_data_model.corners:
    #     if obj is not None:
    #         x1, y1, x2, y2 = obj.x1, obj.y1, obj.x2, obj.y2
    #         label = f"{obj.name} {float(obj.confidence):.2f}"
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         cv2.circle(frame, obj.midpoint, radius=5, color=(0, 0, 255), thickness=-1)
        
    # for obj in [current_data_model.front, current_data_model.back, current_data_model.obstacle, current_data_model.egg, current_data_model.small_goal]:
    #     if obj is not None:
    #         x1, y1, x2, y2 = obj.x1, obj.y1, obj.x2, obj.y2
    #         label = f"{obj.name} {float(obj.confidence):.2f}"
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         cv2.circle(frame, obj.midpoint, radius=5, color=(0, 0, 255), thickness=-1)

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
                    #contour = calculate_mask(frame, new_object)
                    #midpoint = calculate_centroid(contour)
                    #new_object = DetectedObject(name, x1, y1, x2, y2, confidence, midpoint)
                    current_data_model.back = new_object
                case "robot-front":
                    #contour = calculate_mask(frame, new_object)
                    #midpoint = calculate_centroid(contour)
                    #new_object = DetectedObject(name, x1, y1, x2, y2, confidence, midpoint)
                    current_data_model.front = new_object
                case "small-goal":
                    current_data_model.small_goal = new_object
        
def calculate_angle(close_ball):
    if current_data_model.front is None:
        print("Error: 'robot-front' not found in shared_list_copy")
        return None
    
    if current_data_model.back is None:
        print("Error: 'robot-back' not found in shared_list_copy")
        return None
    
    x0, y0 = current_data_model.back.midpoint
    x1, y1 = current_data_model.front.midpoint
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
    
    angle_radians = np.arccos(cos_theta)
    
    if cross_product > 0:
        angle_radians = -angle_radians
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    # Check if angle is within desired range
    return int(angle_degrees)

def calculate_distance(close_ball): 
    if current_data_model.front is None:
        print("Error: 'robot-front' not found in shared_list_copy")
        return None, None
    
    current_x, current_y = current_data_model.front.midpoint
    target_x, target_y = close_ball.midpoint
    
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    distance = math.sqrt(delta_x**2 + delta_y**2)
    
    return int(distance)

def start_client():
    # target_host = "192.168.18.18"
    target_host = "172.20.10.4"
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

def send_command(client_socket, command):
    try:
        client_socket.send(command.encode())
        responce = -1
        response = client_socket.recv(4096)
        while response == -1:
            time.sleep(0.01)
        logging.debug(f"Response command: {response.decode()}")
        return response.decode()
    except Exception as e:
        logging.error(f"An error occurred while sending command: {e}")

def closest_ball():
    closest_distance = float('inf')
    closest_ball = None

    if current_data_model.front:
        for ball in current_data_model.balls:
            distance = np.linalg.norm(np.array(current_data_model.front.midpoint) - np.array(ball.midpoint))
            if distance < closest_distance:
                closest_distance = distance
                closest_ball = ball

    return closest_ball

def move_to_ball(client_socket, frame, close_ball):
    robot_moving = True
    if close_ball is not None:
        angle = calculate_angle(close_ball)
        distance = calculate_distance(close_ball)

        if angle is not None and (-5 > angle or angle > 5):
            command = f"turn{angle}"
        else:
            command = f"drive{distance}"
        logging.debug(f"send command: {command}")
        send_command(client_socket, command)
    robot_moving = False

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

def controller(is_server_online, client_socket):
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

        results = model(frame, verbose=False)
        update_list(results, class_names, frame)
        draw_boxes(frame, results, class_names)
        close_ball = closest_ball()
        if close_ball is None:
            logging.debug("no closest ball found!")
        elif current_data_model.front is None:
            logging.debug("no front found!")
        elif current_data_model.front is None:
            logging.debug("no back found!")
        else:
            cv2.line(frame, (close_ball.x1, close_ball.y1), (current_data_model.front.x1,current_data_model.front.y1), (0, 0, 255), 2)

        cv2.imshow('Frame', frame)
        if True:
            if not is_server_online:
                client_socket = start_client()
                if client_socket:
                    is_server_online = True
                    logging.debug("Started client")
            if client_socket:
                move_to_ball(client_socket, frame, close_ball)
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
