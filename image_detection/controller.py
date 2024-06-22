import threading
import time
import cv2
from queue import Queue
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
    "corner1": 0.60
}

def controller():
    image_thread = threading.Thread(target=image_recognition_thread, args=(model_path, data_yaml_path, video_path, conf_thresholds, shared_list, list_lock, robot_ready, frame_queue))
    robot_thread = threading.Thread(target=robot_client_thread, args=(shared_list, list_lock, robot_ready))
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
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

def get_latest_frame():
    try:
        return frame_queue.get()
    except Exception as e:
        return None

if __name__ == "__main__":
    controller()
