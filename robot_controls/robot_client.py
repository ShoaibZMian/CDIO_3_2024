import sys
import os
import math
import socket

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

robot_moving = False
is_server_online = False
client_socket = None

def calculate_distance_and_angle(target_x, target_y, current_x=0, current_y=0):
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    distance = math.sqrt(delta_x**2 + delta_y**2)
    target_angle = math.degrees(math.atan2(delta_y, delta_x))
    return distance, target_angle

def start_client():
    target_host = "localhost"
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

def move_to_ball(items, client_socket):
    global robot_moving
    robot_moving = True
    send_command(client_socket, "move_to_ball")
    robot_moving = False

def robot_client_thread(shared_list, list_lock, robot_ready):
    global is_server_online, client_socket
    while True:
        with robot_ready:
            robot_ready.wait()
        with list_lock:
            items_to_send = shared_list.copy()
        if not is_server_online:
            client_socket = start_client()
            if client_socket:
                is_server_online = True
                print("Started client")
        if client_socket:
            move_to_ball(items_to_send, client_socket)
