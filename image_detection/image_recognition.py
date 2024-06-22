import cv2
import yaml
import sys
import os
import numpy as np
import threading
import time
from queue import Queue
from ultralytics import YOLO
from itemManager import add_item, get_all_items, reset, update_closest_ball, items_scanned

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
                

def image_recognition_thread(model_path, data_yaml_path, video_path, conf_thresholds, shared_list, list_lock, robot_ready, frame_queue):
    model = YOLO(model_path)
    class_names = load_class_names(data_yaml_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No frame available")
            break

        results = model(frame)
        
        with list_lock:
            update_list(shared_list, results, class_names)

        #draw_boxes(shared_list, frame)
        delay_frame_render("org_frame", frame, 25)

        sorted_corners = sort_corners(shared_list)
        cropped_frame, crop_x_min, crop_y_min = crop_to_corners(
            frame, sorted_corners
        )
        update_corner_coordinates(sorted_corners, crop_x_min, crop_y_min)
        delay_frame_render("cropped_frame", cropped_frame, 25)


        frame_queue.put(frame)

    cap.release()
    cv2.destroyAllWindows()


# Render frame and with delay between each frame
def delay_frame_render(name, frame, delay):
    cv2.imshow(name, frame)
    cv2.waitKey(delay)
    

def crop_to_corners(frame, detected_objects):
    corners = [obj for obj in detected_objects if obj.name.startswith("corner")]

    if not corners:
        return frame, 0, 0  # If no corners are found, return the original frame

    x_min = min([corner.midpoint[0] for corner in corners])
    y_min = min([corner.midpoint[1] for corner in corners])
    x_max = max([corner.midpoint[0] for corner in corners])
    y_max = max([corner.midpoint[1] for corner in corners])

    cropped_frame = frame[y_min:y_max, x_min:x_max]
    return cropped_frame, x_min, y_min


def sort_corners(shared_list):
    sorted_corners = sorted(
        [obj for obj in shared_list if obj.name.startswith("corner")],
        key=lambda obj: (obj.midpoint[0], obj.midpoint[1]),
    )
    for i, obj in enumerate(sorted_corners, start=1):
        obj.name = f"corner{i}"
        print(f"{obj.name}: {obj.midpoint}, {obj.confidence}")
    return sorted_corners


def update_corner_coordinates(corners, crop_x_min, crop_y_min):
    for corner in corners:
        corner.midpoint = (
            corner.midpoint[0] - crop_x_min,
            corner.midpoint[1] - crop_y_min,
        )