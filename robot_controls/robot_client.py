import sys
import os
import math
import socket
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

robot_moving = False
is_server_online = False
client_socket = None

def calculate_angle(close_ball, shared_list_copy):
    robot_front_obj = None
    robot_back_obj = None
    for obj in shared_list_copy:
        if obj.name == "robot-front":
            robot_front_obj = obj
            break
    
    if robot_front_obj is None:
        print("Error: 'robot-front' not found in shared_list_copy")
        return None, None
    
    robot_back_obj = None
    for obj in shared_list_copy:
        if obj.name == "robot-back":
            robot_back_obj = obj
            break
    
    if robot_back_obj is None:
        print("Error: 'robot-back' not found in shared_list_copy")
        return None, None
    
    x0, y0 = robot_back_obj.midpoint
    x1, y1 = robot_front_obj.midpoint
    x2, y2 = close_ball.midpoint

    # Create the vectors from the coordinates
    vector1 = np.array([x1 - x0, y1 - y0])
    vector2 = np.array([x2 - x0, y2 - y0])
    
    # Calculate the dot product
    dot_product = np.dot(vector1, vector2)
    
    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude1 * magnitude2)
    
    # Ensure the value is within the valid range for arccos due to floating point precision issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(cos_theta)
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return int(angle_degrees)

def calculate_distance_and_angle(close_ball, shared_list_copy):
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
    target_angle = math.degrees(math.atan2(delta_y, delta_x))
    
    return distance, target_angle

def start_client():
    target_host = "172.20.10.4"
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

def move_to_ball(shared_list_copy, client_socket):
    global robot_moving
    robot_moving = True
    close_ball = closest_ball(shared_list_copy)
    angel = calculate_angle(close_ball, shared_list_copy)
    command = f"turn{angel}"
    send_command(client_socket, command)
    robot_moving = False

def robot_client_thread(shared_list, list_lock, robot_ready):
    global is_server_online, client_socket
    while True:
        with robot_ready:
            robot_ready.wait()
        with list_lock:
            shared_list_copy = shared_list.copy()
        if not is_server_online:
            client_socket = start_client()
            if client_socket:
                is_server_online = True
                print("Started client")
        if client_socket:
            move_to_ball(shared_list_copy, client_socket)
            client_socket = False
            is_server_online = False
