from ultralytics import YOLO
import cv2
import time
import yaml
import logging

model_path = "/Users/matt/CDIO_3_2024/best.v12/best (1).pt"
data_yaml_path = "/Users/matt/CDIO_3_2024/best.v12/data.yaml"
video_path = 0
conf_thresholds = {'corner1': 0.5, 'egg': 0.5, 'obstacle': 0.5, 'orange-golf-ball': 0.35, 'robot-back': 0.1, 'robot-front': 0.1, 'small-goal': 0.1, 'white-golf-ball': 0.35}

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']   

def draw_boxes(frame, results, class_names):
    for result in results:
        for obj in result.boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            class_index = int(obj.cls)
            name = class_names[class_index]
            confidence = obj.conf
            label = f"{name} {confidence}"
            if name != "robot":
                if name in conf_thresholds and confidence >= conf_thresholds[name]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def controller():
    global phase
    model = YOLO(model_path)
    class_names = load_class_names(data_yaml_path)
    cap = cv2.VideoCapture(video_path)

    time.sleep(2)

    if not cap.isOpened():
            logging.error("Error opening video stream or file")
            return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error("No frame available")
            break

        results = model(frame, verbose=False)

        draw_boxes(frame, results, class_names)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    controller()