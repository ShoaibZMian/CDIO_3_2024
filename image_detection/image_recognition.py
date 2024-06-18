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
    def __init__(self, name, bbox, confidence):
        self.name = name
        self.bbox = bbox  # Bounding box coordinates (x1, y1, x2, y2)
        self.confidence = confidence

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)
    
         

def draw_boxes(detected_objects, frame):
    for obj in detected_objects:
        x1, y1, x2, y2 = obj.bbox
        label = f"{obj.name} {obj.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

import numpy as np  # Import numpy for dtype conversion if needed

def update_list(detected_objects, result, threshold=10):
    if result is not None:
        if not detected_objects:
            # If detected_objects list is empty, add all new detections
            for detection in result:
                bbox = detection.boxes  # Convert to numpy array and then to int
                confidence = detection.confidence
                name = detection.name
                obj = DetectedObject(name, bbox, confidence)
                detected_objects.append(obj)
        else:
            # Update existing objects or add new detections
            updated = False
            for obj in detected_objects:
                for detection in result:
                    bbox = detection.boxes  # Convert to numpy array and then to int
                    confidence = detection.confidence
                    name = detection.name

                    # Check if the detected object matches the existing object within the threshold
                    x1_diff = abs(obj.bbox[0] - bbox[0])
                    y1_diff = abs(obj.bbox[1] - bbox[1])
                    x2_diff = abs(obj.bbox[2] - bbox[2])
                    y2_diff = abs(obj.bbox[3] - bbox[3])

                    if x1_diff <= threshold and y1_diff <= threshold and x2_diff <= threshold and y2_diff <= threshold:
                        # Update the existing object
                        obj.bbox = bbox
                        obj.confidence = confidence
                        updated = True
                        break

                if updated:
                    break
            else:
                # If no match found, add a new detection
                bbox = detection.boxes  # Convert to numpy array and then to int
                confidence = detection.confidence
                name = detection.name
                obj = DetectedObject(name, bbox, confidence)
                detected_objects.append(obj)


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