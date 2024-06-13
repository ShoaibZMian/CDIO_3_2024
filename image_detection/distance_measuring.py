import cv2
import numpy as np
import math

global_distance = 0

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

class TrackedObject:
    def __init__(self, object_type, contour, centroid, circularityShape):
        self.object_type = object_type
        self.contour = contour
        self.centroid = centroid
        self.last_seen = 0  # Frames since last seen'
        self.circularityShape = circularityShape

def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    else:
        return None

def update_tracked_objects(tracked_objects, detected_objects, distance_threshold):
    for detected in detected_objects:
        detected_centroid = detected.centroid
        match_found = False

        for tracked in tracked_objects:
            if tracked.object_type == detected.object_type:
                distance = np.linalg.norm(np.array(tracked.centroid) - np.array(detected_centroid))
                if distance < distance_threshold:
                    tracked.contour = detected.contour
                    tracked.centroid = detected_centroid
                    tracked.last_seen = 0  # Reset last seen counter
                    match_found = True
                    break
        
        if not match_found:
            tracked_objects.append(detected)

    # Increment the last seen counter for all tracked objects
    for tracked in tracked_objects:
        tracked.last_seen += 1

    # Optionally, remove objects that have not been seen for a while (e.g., 10 frames)
    tracked_objects[:] = [obj for obj in tracked_objects if obj.last_seen < 10]

def detect_objects(frame):
    detected_objects = []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges
    lower_white_ball = np.array([0, 0, 190])
    upper_white_ball = np.array([180, 70, 255])
    lower_white_egg = np.array([0, 0, 190])
    upper_white_egg = np.array([180, 70, 255])
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])
    lower_green = np.array([25, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create masks
    mask_white_ball = cv2.inRange(hsv, lower_white_ball, upper_white_ball)
    mask_white_egg = cv2.inRange(hsv, lower_white_egg, upper_white_egg)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_white_ball = cv2.morphologyEx(mask_white_ball, cv2.MORPH_CLOSE, kernel)
    mask_white_ball = cv2.morphologyEx(mask_white_ball, cv2.MORPH_OPEN, kernel)
    mask_white_egg = cv2.morphologyEx(mask_white_egg, cv2.MORPH_CLOSE, kernel)
    mask_white_egg = cv2.morphologyEx(mask_white_egg, cv2.MORPH_OPEN, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    # Find contours for each mask
    contours_white_ball, _ = cv2.findContours(mask_white_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_white_egg, _ = cv2.findContours(mask_white_egg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_purple, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get frame dimensions
    height, width = frame.shape[:2]
    small_corners = []
    for contour in contours_white_ball:
        area = cv2.contourArea(contour)
        if 30 < area < 300:  # Lowered minimum area threshold
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.6 < circularity < 1.2:  # More stringent circularity check
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 1.2 and x > 10 and y > 10 and x + w < width - 10 and y + h < height - 10:
                    centroid = calculate_centroid(contour)
                    if centroid:
                      detected_objects.append(TrackedObject('ball', contour, centroid, circularity))

    
    
    for contour in contours_white_egg:
        area = cv2.contourArea(contour)
        if 300 < area < 1000:
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.4 < circularity < 1.2:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if (aspect_ratio < 0.8 or aspect_ratio > 1.2) and x > 10 and y > 10 and x + w < width - 10 and y + h < height - 10:
                    centroid = calculate_centroid(contour)
                    if centroid:
                        detected_objects.append(TrackedObject('egg', contour, centroid, circularity))

    for contour in contours_orange:
        area = cv2.contourArea(contour)
        if 30 < area < 300:  # Lowered minimum area threshold
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.6 < circularity < 1.2:  # More stringent circularity check
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 1.2 and x > 10 and y > 10 and x + w < width - 10 and y + h < height - 10:
                    centroid = calculate_centroid(contour)
                    if centroid:
                        detected_objects.append(TrackedObject('orange ball', contour, centroid, circularity))

  

    for contour in contours_purple:
        area = cv2.contourArea(contour)
        if area > 0:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.8 < aspect_ratio < 1.2:
                centroid = calculate_centroid(contour)
                if centroid:
                    detected_objects.append(TrackedObject('purple', contour, centroid, None))





    for contour in contours_green:
        area = cv2.contourArea(contour)
        if area > 0:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.8 < aspect_ratio < 1.2:
                centroid = calculate_centroid(contour)
                if centroid:
                    detected_objects.append(TrackedObject('green', contour, centroid, None))
    return detected_objects

    
   

def object_detection_opencv():
    video_path = 0  # Adjust this as needed for your video source
    cap = cv2.VideoCapture(video_path)
    
    tracked_objects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        detected_objects = detect_objects(frame)
        update_tracked_objects(tracked_objects, detected_objects, distance_threshold=20)

        # Draw the tracked objects
        for obj in tracked_objects:
            x, y, w, h = cv2.boundingRect(obj.contour)
            if obj.object_type == 'ball':
                color = (0, 255, 0)  # Green for balls
            elif obj.object_type == 'orange':
                color = (0, 165, 255)  # Orange for orange objects
            elif obj.object_type == 'purple':
                color = (255, 0, 255)  # Purple for purple objects
            elif obj.object_type == 'green':
                color = (0, 255, 255)  # Yellow-green for green objects
            elif obj.object_type == 'egg':
                color = (252, 252, 252)  # White for eggs
        

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            text = f"{obj.object_type}: {id(obj)}: {cv2.contourArea(obj.contour)}:{obj.circularityShape} "
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_position = (x, y - 10)

            # Draw a black rectangle as background for the text
            cv2.rectangle(frame, (text_position[0], text_position[1] - text_height - baseline),(text_position[0] + text_width, text_position[1] + baseline), (0, 0, 0), -1)
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.circle(frame, obj.centroid, 3, (0,0,0), 3)

        cv2.imshow('Tracked Objects', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

object_detection_opencv()
