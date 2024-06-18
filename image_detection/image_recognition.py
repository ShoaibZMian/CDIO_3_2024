import cv2
import yaml
import sys
import os
import numpy as np
import threading
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

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)
    
         

def draw_boxes(detected_objects, frame):
    for obj in detected_objects:
        x1, y1, x2, y2 = obj.x1, obj.y1, obj.x2, obj.y2
        #label = f"{obj.name} {obj.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "test", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

import numpy as np  # Import numpy for dtype conversion if needed

def update_list(detected_objects, results, threshold=10):
    for result in results:
        if result is not None:
            for detected in result.boxes:
                x1, y1, x2, y2 = map(int, detected.xyxy[0])
                name = detected.cls
                confidence = detected.conf
                new_object = DetectedObject(name, x1, y1, x2, y2, confidence)
                detected_objects.append(new_object)
        else:
            for detected in result.boxes:
                x1, y1, x2, y2 = map(int, detected.xyxy[0])
                name = detected.cls
                confidence = detected.conf
                
                updated = False
                for existing in detected_objects:
                    # Check if the detected object matches the existing object within the threshold
                    x1_diff = abs(existing.x1 - x1)
                    y1_diff = abs(existing.y1 - y1)
                    x2_diff = abs(existing.x2 - x2)
                    y2_diff = abs(existing.y2 - y2)

                    if x1_diff <= threshold and y1_diff <= threshold and x2_diff <= threshold and y2_diff <= threshold:
                        # Update the existing object
                        existing.x1, existing.y1, existing.x2, existing.y2 = x1, y1, x2, y2
                        existing.confidence = confidence
                        updated = True
                        break

                if not updated:
                    # If no match found, add a new detection
                    new_object = DetectedObject(name, x1, y1, x2, y2, confidence)
                    detected_objects.append(new_object)


def image_recognition_thread(model_path, data_yaml_path, video_path, conf_thresholds, shared_list, list_lock, robot_ready, frame_queue):
    model = YOLO(model_path)
    data = load_yaml(data_yaml_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No frame available")
            break

        results = model(frame)
        
        for result in results:
            update_list(shared_list, result)

        draw_boxes(shared_list, frame)

        # Put the annotated frame into the queue
        frame_queue.put(frame)

    cap.release()
    cv2.destroyAllWindows()