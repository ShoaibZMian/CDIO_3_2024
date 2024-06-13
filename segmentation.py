import cv2
import numpy as np
from image_data import ProcessedImageData, TrackedObject


def closest_object(processed_image):
    closest_distance = float('inf')
    closest_ball = None

    for green_object in processed_image:
        if green_object.object_type == "green":
            for ball_object in processed_image:
                if ball_object.object_type == "ball":
                    distance = np.linalg.norm(np.array(green_object.centroid) - np.array(ball_object.centroid))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_ball = ball_object

    return closest_ball

def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    else:
        return None

def update_processed_image(processed_image, detected_objects, distance_threshold):
    for detected in detected_objects:
        detected_centroid = detected.centroid
        match_found = False

        for tracked in processed_image:
            if tracked.object_type == detected.object_type:
                distance = np.linalg.norm(np.array(tracked.centroid) - np.array(detected_centroid))
                if distance < distance_threshold:
                    tracked.contour = detected.contour
                    tracked.centroid = detected_centroid
                    tracked.last_seen = 0  # Reset last seen counter
                    match_found = True
                    break
        
        if not match_found:
            processed_image.append(detected)

    # Increment the last seen counter for all tracked objects
    for tracked in processed_image:
        tracked.last_seen += 1

    # Optionally, remove objects that have not been seen for a while (e.g., 10 frames)
    processed_image[:] = [obj for obj in processed_image if obj.last_seen < 10]

def detect_objects(frame):
    detected_objects = []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges
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
    
    # Create masks
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    # Find contours for each mask
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_purple, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_white:
        area = cv2.contourArea(contour)
        if 450 < area < 1000:
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.4 < circularity < 1.2:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 1.2:
                    centroid = calculate_centroid(contour)
                    if centroid:
                        detected_objects.append(TrackedObject('ball', contour, centroid))

    for contour in contours_purple:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.8 < aspect_ratio < 1.2:
                centroid = calculate_centroid(contour)
                if centroid:
                    detected_objects.append(TrackedObject('purple', contour, centroid))

    for contour in contours_green:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.8 < aspect_ratio < 1.2:
                centroid = calculate_centroid(contour)
                if centroid:
                    detected_objects.append(TrackedObject('green', contour, centroid))

    return detected_objects

def object_detection_opencv(shared_processed_image, condition):
    video_path = 0  # Adjust this as needed for your video source
    cap = cv2.VideoCapture(video_path)
    
    image_id: int = 1
    result: ProcessedImageData = shared_processed_image
    tracked_objects: list[TrackedObject] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        detected_objects = detect_objects(frame)
        update_processed_image(tracked_objects, detected_objects, distance_threshold=20)

        # Draw the tracked objects
        for obj in tracked_objects:
            x, y, w, h = cv2.boundingRect(obj.contour)
            if obj.object_type == 'ball':
                color = (0, 255, 0)  # Green for balls
            elif obj.object_type == 'purple':
                color = (255, 0, 255)  # Purple for purple objects
            elif obj.object_type == 'green':
                color = (0, 255, 255)  # Yellow-green for green objects

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            text = f"{obj.object_type}: {id(obj)}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_position = (x, y - 10)

            # Draw a black rectangle as background for the text
            cv2.rectangle(frame, (text_position[0], text_position[1] - text_height - baseline),(text_position[0] + text_width, text_position[1] + baseline), (0, 0, 0), -1)
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.circle(frame, obj.centroid, 3, (0,0,0), 3)

        with condition:
            result.id = image_id
            result.is_fresh = True
            result.tracked_objects = tracked_objects

        image_id += 1
        print("before")
        cv2.imshow('Tracked Objects', frame)
        print("after")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()