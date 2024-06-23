import threading
import time
import cv2
import numpy as np
from queue import Queue
from matplotlib import pyplot as plt
from image_recognition import image_recognition_thread
from robot_controls.robot_client import robot_client_thread

shared_list = []
list_lock = threading.Lock()
robot_ready = threading.Condition()
frame_queue = Queue()

model_path = "C:/Users/Shweb/Downloads/v13/best.pt"
data_yaml_path = "C:/Users/Shweb/Downloads/cdio3.v13i.yolov8/data.yaml"
video_path = 1
conf_thresholds = {
    'white-golf-ball': 0.4,
    'robot-front': 0.25,
    'robot-back': 0.25,
    'egg': 0.70,
    'corner1': 0.20,
    'obstacle': 0.20,
}


def controller():
    image_thread = threading.Thread(target=image_recognition_thread, args=(
        model_path, data_yaml_path, video_path, conf_thresholds, shared_list, list_lock, robot_ready, frame_queue))
    robot_thread = threading.Thread(
        target=robot_client_thread, args=(shared_list, list_lock, robot_ready))
    image_thread.start()
    robot_thread.start()

    while True:
        with list_lock:
            if shared_list:
                with robot_ready:
                    robot_ready.notify()

        frame = get_latest_frame()
        if frame is not None:
            cv2.imshow('Final frame', frame)

            grid = create_grid(frame, shared_list)
            display_grid_on_image(frame, grid)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def get_latest_frame():
    try:
        return frame_queue.get()
    except Exception as e:
        return None


# Grid values
# • 0 for empty cells
# • 1 for obstacles
# • 2 for the start point
# • 3 for the end point
class Item:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

def create_grid(frame, items):
    # Convert frame to RGB
    frame_to_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_size_y, frame_size_x, _ = frame_to_img.shape

    # Calculate grid size
    grid_size_x = frame_size_x
    grid_size_y = frame_size_y

    # Debug print statements
    print(f"Frame size: {frame_size_x} x {frame_size_y}")
    print(f"Grid size: {grid_size_x} x {grid_size_y}")

    # Create a grid with specified dimensions
    grid = np.zeros((grid_size_y, grid_size_x))

    # Calculate scaling factors to translate from frame coordinates to grid coordinates
    scale_x = grid_size_x / frame_size_x
    scale_y = grid_size_y / frame_size_y

    # Debug print statements
    print(f"Scaling factors - X: {scale_x}, Y: {scale_y}")

    # Place items on the grid
    for item in items:
        grid_x1 = int(item.x1 * scale_x)
        grid_x2 = int(item.x2 * scale_x)
        grid_y1 = int(item.y1 * scale_y)
        grid_y2 = int(item.y2 * scale_y)
        
        # Debug print statements
        print(f"Item coordinates (frame): ({item.x1}, {item.y1}), ({item.x2}, {item.y2})")
        print(f"Item coordinates (grid): ({grid_x1}, {grid_y1}), ({grid_x2}, {grid_y2})")
        
        for x in range(grid_x1, grid_x2 + 1):
            for y in range(grid_y1, grid_y2 + 1):
                if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
                    grid[y][x] = 1  # Note: grid coordinates (y, x)

    return grid

def display_grid_on_image(frame, grid):
    plt.figure(figsize=(10, 10))
    plt.imshow(frame)
    plt.imshow(grid, cmap='hot', alpha=0.5)  # Overlay grid with some transparency
    plt.gca()  # Invert y-axis to match the image coordinates
    plt.show()

if __name__ == "__main__":
    controller()

























