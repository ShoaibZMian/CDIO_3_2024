import threading
import time
import cv2
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from typing import Optional, Tuple, Set
from image_recognition import image_recognition_thread
from robot_controls.robot_client import robot_client_thread

shared_list = []
list_lock = threading.Lock()
robot_ready = threading.Condition()
frame_queue = Queue()

model_path = "/home/shweb/Documents/best_pt/bestv14/v14bestL/best.pt"
data_yaml_path = "/home/shweb/Documents/dataset/cdio3.v14i/data.yaml"
video_path = 4
conf_thresholds = {
    'white-golf-ball': 0.4,
    'robot-front': 0.25,
    'robot-back': 0.25,
    #'egg': 0.50,
    'corner1': 0.20,
    'obstacle': 0.20,
    #'small-goal': 0.20,
    #'orange-golf-ball': 0.35,
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
            # display_grid_on_image(frame, grid) does not refresh the image

            steps_needed = bfs_shortest_path(grid)
            print("Least amount of steps needed:", steps_needed)

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
    def __init__(self, name, x1, y1, x2, y2):
        self.name = name
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

    # Create a grid with specified dimensions
    grid = np.zeros((grid_size_y, grid_size_x))

    # Calculate scaling factors to translate from frame coordinates to grid coordinates
    scale_x = grid_size_x / frame_size_x
    scale_y = grid_size_y / frame_size_y

    # Place items on the grid
    for item in items:
        grid_x1 = int(item.x1 * scale_x)
        grid_x2 = int(item.x2 * scale_x)
        grid_y1 = int(item.y1 * scale_y)
        grid_y2 = int(item.y2 * scale_y)

        # Set grid value based on item name
        if item.name == 'white-golf-ball':
            grid_value = 3
        elif item.name == 'robot-front':
            grid_value = 2
        elif item.name == 'obstacle' or item.name == 'egg':
            grid_value = 1
        else:
            grid_value = 0  # Default value for other items (Show as empty cell)

        for x in range(grid_x1, grid_x2 + 1):
            for y in range(grid_y1, grid_y2 + 1):
                if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
                    grid[y][x] = grid_value  # Note: grid coordinates (y, x)

    return grid


def display_grid_on_image(frame, grid):
    fig, ax = plt.subplots(figsize=(10, 10))

    def update(*args):
        ax.clear()
        image_frame = ax.imshow(frame)
        image_grid = ax.imshow(grid, cmap='hot', alpha=0.5)
        return [image_frame, image_grid]  # Return the list of artists

    ani = animation.FuncAnimation(fig, update, interval=1000)  # Update every 100 milliseconds
    plt.show()



def bfs_shortest_path(grid: np.ndarray) -> int:
    # Define the possible movements (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows, cols = grid.shape
    start: Optional[Tuple[int, int]] = None
    end: Optional[Tuple[int, int]] = None

    # Locate the start and end points in the grid
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 2:
                start = (r, c)
            elif grid[r, c] == 3:
                end = (r, c)

    if start is None or end is None:
        return -1  # Return -1 if start or end point is not found

    # Initialize the BFS queue and visited set
    queue: deque[Tuple[Tuple[int, int], int]] = deque([((start[0], start[1]), 0)])  # (current position, steps taken)
    visited: Set[Tuple[int, int]] = set([start])

    while queue:
        (current_row, current_col), steps = queue.popleft()

        # Check if we have reached the end point
        if (current_row, current_col) == end:
            return steps

        # Explore all possible directions
        for dr, dc in directions:
            new_row, new_col = current_row + dr, current_col + dc

            # Check if the new position is within bounds and not an obstacle or visited
            if 0 <= new_row < rows and 0 <= new_col < cols and grid[new_row, new_col] != 1 and (new_row, new_col) not in visited:
                visited.add((new_row, new_col))
                queue.append(((new_row, new_col), steps + 1))

    return -1  # Return -1 if no path is found

if __name__ == "__main__":
    controller()

























