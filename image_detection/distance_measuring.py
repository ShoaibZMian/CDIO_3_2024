from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
import torch

global_distance = 0


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
        "C:/Users/Shweb/Downloads/train4-20240608T153413Z-001.zip/train4/weights/best.pt"
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
    object_width_pixels = np.sqrt(np.power((p1[0] - p2[0]), 2) + np.power((p1[1] - p2[1]), 2))

    pixel_size_meters = distance_between / object_width_pixels

    global global_distance
    global_distance = pixel_size_meters


def calculate_distance(p1, p2, pixel_size_meters):
    distance_in_pixels = np.sqrt(np.power((p1[0] - p2[0]), 2) + np.power((p1[1] - p2[1]), 2))
    result = distance_in_pixels * pixel_size_meters
    rounded_result = np.round(result, 3)
    return rounded_result


def video_object_tracking():
    model = YOLO(
        "C:/Users/Shweb/Downloads/train4-20240608T153413Z-001/train4/weights/best.pt"
    )

    video_path = (
        "C:/Users/Shweb/Downloads/Filmm.mov"
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
    model = YOLO("/Users/matt/Downloads/run_v9_n_50e_8b/weights/best.pt")
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model.to(device)

    video_path = (1)
    cap = cv2.VideoCapture(video_path)

    class_names = ['corner1', 'egg', 'obstacle', 'orange-golf-ball', 'robot', 'robot-back', 'robot-front', 'small-goal', 'white-golf-ball']

    ret = True
    while ret:
        ret, frame = cap.read()

        if ret:
            results = model.track(frame, persist=True, conf=0.5, iou=0.3)

            #frame_ = results[0].plot()
            #cv2.putText(frame, f"Model is using: {device}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            corners = 0
            corner1_cord = 0
            corner2_cord = 0
            for result in results:
                if result.boxes is not None:
                    for obj in result.boxes:
                        class_name = class_names[int(obj.cls[0])]

                        if class_names == "corner1":
                                cv2.putText(frame, "+ Corner", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                corners +=1
                                if corners == 1:
                                    corner1_cord = obj.xyxy[0]
                                else:
                                    corner2_cord = obj.xyxy[0]

                        if corners == 2:
                            cv2.putText(frame, "Mapping", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            calculate_mapping(corner1_cord, corner2_cord, 180)
                        
                        x1, y1, x2, y2 = map(int, obj.xyxy[0])
                        label = obj.cls
                        confidence = obj.conf

                        cX = (x1 + x2) // 2
                        cY = (y1 + y2) // 2
                        #text = f'Name: {class_name} Conf: {confidence} X={cX}, Y={cY}'
                        text = f'Map: {global_distance} Name: {class_name} X={cX}, Y={cY}'
                        text_position = (cX, cY - 10)

                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (text_position[0], text_position[1] - text_height - baseline),
                                        (text_position[0] + text_width, text_position[1] + baseline), (0, 0, 0), -1)

                        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            cv2.imshow("frame", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

def calculate_triangle_mid(contours):
    if not contours:
        raise ValueError("Contours are empty")
        return None
    threshold = 20
    max_x = float('-inf')
    min_x = float('inf')
    max_y = float('-inf')
    min_y = float('inf')

    max_x_coord = (0, 0)
    min_x_coord = (0, 0)
    max_y_coord = (0, 0)
    min_y_coord = (0, 0)

    # Iterate through all contours
    for contour in contours:
        # Iterate through all points in the contour
        for point in contour:
            # Get the x and y coordinates
            x, y = point[0]  # point[0] gives the (x, y) coordinates
            
            # Update the max and min x values and their coordinates
            if x > max_x:
                max_x = x
                max_x_coord = (x, y)
            if x < min_x:
                min_x = x
                min_x_coord = (x, y)

            # Update the max and min y values and their coordinates
            if y > max_y:
                max_y = y
                max_y_coord = (x, y)
            if y < min_y:
                min_y = y
                min_y_coord = (x, y)

    # Collect all four coordinates
    coords = [max_x_coord, min_x_coord, max_y_coord, min_y_coord]

    # List to hold pairs of points that are close to each other
    close_pairs = []

    # Find pairs of points that are close to each other
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            # Calculate the Euclidean distance directly
            distance = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
            close_pairs.append((coords[i], coords[j], distance))

    # Find the closest pair of coordinates
    if close_pairs:
        closest_pair = min(close_pairs, key=lambda x: x[2])  # Find the pair with the smallest distance
        coord_to_remove = closest_pair[1]  # Choose to remove the second coordinate in the closest pair
        final_coords = [coord for coord in coords if coord != coord_to_remove]
    else:
        final_coords = coords

    # Calculate the midpoint of the remaining three coordinates (centroid)
    centroid_x = np.mean([coord[0] for coord in final_coords])
    centroid_y = np.mean([coord[1] for coord in final_coords])

    return (int(centroid_x), int(centroid_y))


def object_detection_opencv():
    video_path = (1)
    cap = cv2.VideoCapture(video_path)

    ball_id = 1
    border_id = 1
    purple_id = 1
    green_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define HSV range
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 70, 255])
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([160, 255, 255])
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        # Create a masks
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)
        mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

        # Find contours on the masked image
        contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_purple, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_white:
            area = cv2.contourArea(contour)
            
            if 450 < area < 1000:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if 0.4 < circularity < 1.2:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.8 < aspect_ratio < 1.2:
                        x_c = int(x+(w/2))
                        y_c = int(y+(h/2))
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.circle(frame, (x_c, y_c), 5, (0, 0, 255), -1)

                        text = f" X: {x_c} Y: {y_c} Ball {ball_id}: Area: {area:.2f}, Circ: {circularity:.2f}, AR: {aspect_ratio:.2f}"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        ball_id += 1

        purple_mid_cord = -1
        if contours_purple:
            purple_mid_cord = calculate_triangle_mid(contours_purple)
        
        for contour in contours_purple:
            area = cv2.contourArea(contour)
            
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                if 0.8 < aspect_ratio < 1.2:
                    cv2.drawContours(frame, [contour], -1, (255, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.circle(frame, purple_mid_cord, 3, (0, 0, 255), -1)

                    text = f"Purple {purple_id}: Area: {area:.2f}, AR: {aspect_ratio:.2f}"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    purple_id += 1

        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        green_mid_cord = -1
        if contours_green:
            green_mid_cord = calculate_triangle_mid(contours_green)

        for contour in contours_green:
            area = cv2.contourArea(contour)
            
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                if 0.8 < aspect_ratio < 1.2:
                    cv2.drawContours(frame, [contour], -1, (255, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                    cv2.circle(frame, green_mid_cord, 3, (0, 0, 255), -1)

                    text = f"Green {purple_id}: Area: {area:.2f}, AR: {aspect_ratio:.2f}"
                    text_position = (x, y - 10)
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (text_position[0], text_position[1] - text_height - baseline), (text_position[0] + text_width, text_position[1] + baseline), (0, 0, 0), -1)
                    cv2.putText(frame, text, (text_position[0],text_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    green_id += 1

        if purple_mid_cord and green_mid_cord != -1:
                    cv2.line(frame, purple_mid_cord, green_mid_cord, (0, 0, 0), 3)

        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#video_object_tracking_gpu()
#object_detection_opencv()
#video_object_tracking_gpu()
object_detection_opencv()