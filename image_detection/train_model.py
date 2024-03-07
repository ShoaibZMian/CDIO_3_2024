from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import cv2

def train_model():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())


    if __name__ == '__main__':
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="C:/Users/SkumJustEatMe/Image_Analyse/roboflow/data.yaml", device=0, epochs=100, imgsz=640)

def run_model_with_tracker():

    weights = 'C:/Users/SkumJustEatMe/Image_Analyse/runs/detect/train9/weights/best.pt'  # Path to pre-trained weights
    model = YOLO(weights)

    # Process video
    video_path = 'C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/WIN_20240228_11_59_40_Pro.mp4'
    output_path = './output_video.avi'
    cap = cv2.VideoCapture(video_path)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))
    tracker = DeepSort(max_age=50)

    #Initiate the varibles i need
    CONFIDENCE_THRESHOLD = 0.6
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = model(frame)[0]

        results = []

        for data in detections.boxes.data.tolist():
            confidence = data[4]

            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])


        # # # # for data in results.boxes.data.tolist():
        # # # #         # extract the confidence (i.e., probability) associated with the detection
        # # # #         confidence = data[4]

        # # # #         if float(confidence) < CONFIDENCE_THRESHOLD:
        # # # #             continue

        # # # #         xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        # # # #         cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)



        tracks = tracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)


        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

run_model_with_tracker