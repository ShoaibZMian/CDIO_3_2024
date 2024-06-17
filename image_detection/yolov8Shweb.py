import cv2
import yaml
import sys
import os
from ultralytics import YOLO
from itemManager import add_item, get_all_items, reset, update_closest_ball, items_scanned

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from robot_controls.robotManager import robot_process_items


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def calculate_center(x1, y1, x2, y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)

def main(model_path, data_yaml_path, video_path):
    # Load model and data
    model = YOLO(model_path)
    data = load_yaml(data_yaml_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while cap.isOpened():
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

                # Calculate center point
                center_x, center_y = calculate_center(x1, y1, x2, y2)

                # Draw bounding box and center point
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Print the center point coordinates
                print(f"Object class: {data['names'][cls]}, Confidence: {conf:.2f}, Center: ({center_x}, {center_y})")
                add_item(data['names'][cls], center_x, center_y)
        
        items_scanned()
        update_closest_ball()
        items = get_all_items()
        print("Items ")
        print(items) 

        robot_process_items(items)

        reset()
        # Display the frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exits
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "C:/Users/Shweb/Downloads/best.v12/best.v12/best.pt"
    data_yaml_path = "C:/Users/Shweb/Downloads/cdio3.v12i.yolov8/data.yaml"
    video_path = 1  # Or use 0 for webcam
    main(model_path, data_yaml_path, video_path)
