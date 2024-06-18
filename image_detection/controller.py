import threading
import time
import cv2
from queue import Queue
from image_recognition import image_recognition_thread
from robot_controls.robot_client import robot_client_thread

shared_list = []
list_lock = threading.Lock()
robot_ready = threading.Condition()
frame_queue = Queue()  # Queue for storing frames

model_path = "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/dataset_v12/best.pt"
data_yaml_path = "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/dataset_v12/data.yaml"
video_path = "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/test_video.mp4"
conf_thresholds = {
    'white-golf-ball': 0.4,
    'robot-front': 0.25,
    'robot-back': 0.25,
    'egg': 0.70,
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
        
        # Get the latest frame from the queue
        frame = get_latest_frame()
        if frame is not None:
            cv2.imshow('Frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        time.sleep(0.1)  # Adjust sleep time as necessary

def get_latest_frame():
    try:
        # Get the latest frame from the queue
        return frame_queue.get_nowait()
    except Exception as e:
        return None

if __name__ == "__main__":
    controller()
