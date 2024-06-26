import cv2
import yaml
import sys
import os
import numpy as np
import threading
import time
from queue import Queue
from ultralytics import YOLO
from image_detection.Clean_Up.itemManager import add_item, get_all_items, reset, update_closest_ball, items_scanned

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

def calculate_midpoint(x1, y1, x2, y2):
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

def update_list(detected_objects, results, class_names, conf_thresholds):
    detected_objects.clear()
    
    for result in results:
        for detected in result.boxes:
            x1, y1, x2, y2 = map(int, detected.xyxy[0])
            class_index = int(detected.cls)
            name = class_names[class_index]
            confidence = detected.conf
            
            if name in conf_thresholds and confidence >= conf_thresholds[name]:
                new_object = DetectedObject(name, x1, y1, x2, y2, confidence)
                detected_objects.append(new_object)
                print(f"Detected {new_object.name} with confidence {new_object.confidence}")

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

        results = model(frame, verbose=False)
        
        with list_lock:
            update_list(shared_list, results, class_names, conf_thresholds)

        sorted_corners = sort_corners(shared_list)
        cropped_frame, crop_x_min, crop_y_min = crop_to_corners(frame, sorted_corners)
        update_corner_coordinates(sorted_corners, crop_x_min, crop_y_min)

        fixed_frame = stretch_corners_to_frame(cropped_frame, sorted_corners)
        
        if fixed_frame.size == 0:
            print("Fixed frame is empty, skipping.")
            continue
        
        print(f"Fixed frame shape: {fixed_frame.shape}")

        results = model(fixed_frame)
        
        with list_lock:
            update_list(shared_list, results, class_names, conf_thresholds)

        draw_boxes(shared_list, fixed_frame)
        frame_queue.put(fixed_frame)

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

def stretch_corners_to_frame(cropped_frame, detected_objects):
    # Get cropped_frame dimensions
    orig_height, orig_width = cropped_frame.shape[:2]

    # Find specific corners by name
    corner1 = next(
        (corner for corner in detected_objects if corner.name == "corner1"), None
    )
    corner2 = next(
        (corner for corner in detected_objects if corner.name == "corner2"), None
    )
    corner3 = next(
        (corner for corner in detected_objects if corner.name == "corner3"), None
    )
    corner4 = next(
        (corner for corner in detected_objects if corner.name == "corner4"), None
    )

    if not all([corner1, corner2, corner3, corner4]):
        print("Not all corners detected.")
        return cropped_frame  # Need all 4 corners to perform the transformation

    # Extract corner points
    corners = [
        corner1.midpoint, # type: ignore
        corner2.midpoint, # type: ignore
        corner3.midpoint, # type: ignore
        corner4.midpoint, # type: ignore
    ]

    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = sorted(corners, key=lambda p: (p[1], p[0]))
    if corners[0][0] > corners[1][0]:
        corners[0], corners[1] = corners[1], corners[0]
    if corners[2][0] < corners[3][0]:
        corners[2], corners[3] = corners[3], corners[2]

    # Visualize the detected and ordered corners
    for i, point in enumerate(corners):
        cv2.circle(cropped_frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        cv2.putText(
            cropped_frame,
            f"Corner {i+1}",
            (int(point[0]), int(point[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    src = np.array(corners, dtype="float32")

    # Define the destination points, the corners of the frame
    dst = np.array(
        [
            [0, 0],
            [orig_width - 1, 0],
            [orig_width - 1, orig_height - 1],
            [0, orig_height - 1],
        ],
        dtype="float32",
    )

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(cropped_frame, M, (orig_width, orig_height))

    return warped
