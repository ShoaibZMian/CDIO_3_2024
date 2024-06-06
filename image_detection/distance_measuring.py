from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
import torch


def show_img():
    img_path = "/Users/matt/CDIO_3_2024/image_detection/data/small dataset/train/images/WIN_20240313_09_24_04_Pro_mp4-0002_jpg.rf.fb71d3f1dd2e7327ede6b41f3be4e43a.jpg"
    img = cv2.imread(img_path)

    circles = np.array([[536, 31], [542, 581]])

    radius = 5

    cv2.circle(
        img=img, center=circles[0], radius=radius, color=(255, 0, 0), thickness=-1
    )
    cv2.putText(
        img=img,
        text="text",
        org=(circles[0, 0] - radius, circles[0, 1] - radius - 5),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(255, 0, 0),
    )

    cv2.imshow("Image with Circles and Labels", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_video(video_path, mp, obj1, obj2):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        one_pixel = calculate_mapping(mp[0], mp[1], 180)

        cv2.putText(
            frame,
            f"Fps: {str(int(cap.get(cv2.CAP_PROP_FPS)))}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"1 Pixel: {str(np.round(one_pixel, 4))}cm",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        for point in mp:
            center_x, center_y = point
            cv2.circle(
                img=frame,
                center=(center_x, center_y),
                radius=5,
                color=(255, 0, 0),
                thickness=-1,
            )
        cv2.line(frame, tuple(mp[0]), tuple(mp[1]), (255, 0, 0), thickness=2)
        midpoint = ((mp[0, 0] + mp[1, 0]) // 2, ((mp[0, 1] + mp[1, 1]) // 2) - 5)
        cv2.putText(
            frame, "180 cm", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
        )

        cv2.circle(
            img=frame,
            center=(obj1[0], obj1[1]),
            radius=5,
            color=(255, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            frame, "Ball 1", obj1 - 5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )
        cv2.circle(
            img=frame,
            center=(obj2[0], obj2[1]),
            radius=5,
            color=(255, 0, 0),
            thickness=-2,
        )
        cv2.putText(
            frame, "Ball 1", obj2 + 5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )
        cv2.line(frame, obj1, obj2, (255, 0, 0), thickness=2)
        midpoint2 = ((obj1[0] + obj2[0]) // 2, ((obj1[1] + obj2[1]) // 2) - 5)
        cv2.putText(
            frame,
            f"{str(np.round(calculate_distance(obj1, obj2, one_pixel), 4))}cm",
            midpoint2,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Object Detection", frame)

        key = cv2.waitKey(1)
        if key == 27:  # 27 = esc
            break

    cap.release()
    cv2.destroyAllWindows()


def show_webcam():
    model = YOLO(
        "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/dataset_v1/runs/detect/yolov8n_b8_50e/weights/best.pt"
    )
    classNames = [
        "BACK",
        "BALL",
        "BIG_GOAL",
        "BORDERS",
        "EGG",
        "FRONT",
        "OBSTACLES",
        "ROBOT",
        "SMALL_GOAL",
    ]

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    prev_time = time.time()

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )  # convert to int values

                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2

                cv2.circle(
                    img=img,
                    center=(mid_x, mid_y),
                    radius=5,
                    color=(255, 0, 0),
                    thickness=-2,
                )
                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                # print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                # print(f"Class name --> {classNames[cls]}")

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (255, 0, 0)
                thickness = 1

                text = f"{classNames[cls]}: {confidence}: x={mid_x} y={mid_y}"
                cv2.putText(img, text, org, font, fontScale, color, thickness)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Display FPS
        cv2.putText(
            img,
            f"Fps: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def calculate_mapping(p1, p2, distance_between):
    object_width_pixels = np.sqrt(
        np.power((p1[0] - p2[0]), 2) + np.power((p1[1] - p2[1]), 2)
    )

    pixel_size_meters = distance_between / object_width_pixels

    return pixel_size_meters


def calculate_distance(p1, p2, pixel_size_meters):
    distance_in_pixels = np.sqrt(
        np.power((p1[0] - p2[0]), 2) + np.power((p1[1] - p2[1]), 2)
    )
    result = distance_in_pixels * pixel_size_meters
    rounded_result = np.round(result, 3)
    return rounded_result


def video_object_tracking():
    model = YOLO(
        "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/dataset_v2/runs/detect/yolov8m_b8_50e/weights/best.pt"
    )

    video_path = (
        "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/test_video.mp4"
    )
    cap = cv2.VideoCapture(video_path)

    ret = True
    while ret:
        ret, frame = cap.read()

        if ret:

            results = model.track(frame, persist=True, conf=0.6, iou=0.3)
            frame_ = results[0].plot()

            # visualize
            cv2.imshow("frame", frame_)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break


def video_object_tracking_gpu():
    # Load model and move to GPU
    model = YOLO(
        "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/dataset_v2/runs/detect/yolov8m_b8_50e/weights/best.pt"
    )
    model.to("cuda")  # Move model to GPU

    video_path = (
        "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/test_video.mp4"
    )
    cap = cv2.VideoCapture(video_path)

    ret = True
    # read frames
    while ret:
        ret, frame = cap.read()

        if ret:
            results = model.track(frame, persist=True, conf=0.5, iou=0.3)

            #frame_ = results[0].plot()

            # Extract coordinates of detected objects
            for result in results:
                for obj in result.boxes:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    label = obj.cls  # Class ID of the detected object
                    confidence = obj.conf  # Confidence of the detection

                    cX = (x1 + x2) // 2
                    cY = (y1 + y2) // 2
                    text = f'Conf: {confidence} Lab: {label} X={cX}, Y={cY}'
                    text_position = (cX, cY - 10)

                    # Draw a rectangle as background for better text visibility
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (text_position[0], text_position[1] - text_height - baseline),
                                    (text_position[0] + text_width, text_position[1] + baseline), (0, 0, 0), -1)

                    # Draw coordinates on the frame
                    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            # Visualize
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# measurement_point1 = np.array([366, 52])
# measurement_point2 = np.array([1642, 49])

# ball1 = np.array([388, 839])
# ball2 = np.array([893, 959])

# measurement_points = np.array([
#     measurement_point1,
#     measurement_point2
# ])

# show_video("/Users/matt/CDIO_3_2024/image_detection/data/test_video.mp4", measurement_points, ball1, ball2)
# print(torch.cuda.is_available())
video_object_tracking_gpu()
