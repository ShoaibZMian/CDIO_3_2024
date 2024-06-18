import cv2
import yaml
import sys
import os
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_detection.itemManager import add_item, get_all_items, reset, update_closest_ball, items_scanned, is_reset
from robot_controls.robotManager import robot_process_items

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def calculate_center(x1, y1, x2, y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)

def yolo_init(model_path, data_yaml_path, video_path, conf_thresholds):
    # Load model and data
    model = YOLO(model_path)
    data = load_yaml(data_yaml_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while cap.isOpened():
        if is_reset():  
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            results = model(frame)
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                classes = result.boxes.cls.cpu().numpy()  # class indices
                confidences = result.boxes.conf.cpu().numpy()  # confidences

                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    cls = int(classes[i])
                    conf = confidences[i]

                    # Apply confidence threshold for the specific class
                    item_name = data['names'][cls]
                    conf_threshold = conf_thresholds.get(item_name, 0.25)  # Default threshold if not specified
                    if conf < conf_threshold:
                        continue

                    # Calculate center point
                    center_x, center_y = calculate_center(x1, y1, x2, y2)

                    # Draw bounding box and center point
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Print the center point coordinates
                    print(f"Object class: {item_name}, Confidence: {conf:.2f}, Center: ({center_x}, {center_y})")
                    add_item(item_name, center_x, center_y)
            
            items_scanned()
            #update_closest_ball()
        
            #items = get_all_items()
            #print("Items ")
            #print(items) 

            # call this from client
            #robot_process_items(items)

        
        # Display the frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_yolo():
    model_path = "C:/Users/Shweb/Downloads/best.v12/best.v12/best.pt"
    data_yaml_path = "C:/Users/Shweb/Downloads/cdio3.v12i.yolov8/data.yaml"
    video_path = 1  # webcam 0 or 1 or "video file path"

    # Define confidence thresholds for specific items
    conf_thresholds = {
        'white-golf-ball': 0.4,  
        'robot-front': 0.25,     
        'robot-back': 0.25,      
        'egg': 0.70, 
    }
    yolo_init(model_path, data_yaml_path, video_path, conf_thresholds)

