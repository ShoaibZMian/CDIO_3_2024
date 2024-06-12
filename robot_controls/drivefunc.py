from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait
from pybricks.robotics import DriveBase
from pybricks.ev3devices import GyroSensor
from pybricks.parameters import Direction
import math
from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
import torch

ev3 = EV3Brick()

# Tilslut gyrosensoren til port 1
gyro = GyroSensor(Port.S1)
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

# Constants
WHEEL_DIAMETER = 56  # Hjulets størrelse i mm
AXEL_TRACK = 145  # Distancen mellem hjulene

robot = DriveBase(left_motor, right_motor, WHEEL_DIAMETER, AXEL_TRACK)

def drive_forward(distance_mm):
    gyro.reset_angle(0)
    robot.reset()
    distance_cm = distance_mm * 100

    while robot.distance() < distance_cm:
        angle_error = gyro.angle()  # Negative feedback for correction
        correction = 6 * angle_error  # Apply proportional control to the correction

        robot.drive(250, correction)# hastigheden må ikke ændres, da længden vil blive upræcis

    robot.stop()
    left_motor.brake()
    right_motor.brake()

    final_angle = gyro.angle()
    text = "Final angle deviation: {} degrees".format(final_angle)
    return text

def drive_backward(distance_mm):
    while robot.distance() > -distance_mm:
        robot.drive(-400, 0)
    robot.stop()
    left_motor.brake()
    right_motor.brake()
    return "Driven backward "+ distance_mm +"mm"

def turn_left(degrees):
    gyro.reset_angle(0)
    initial_angle = gyro.angle()
    target_angle = initial_angle - degrees
    while gyro.angle() > target_angle:
        robot.drive(0, 80)
    robot.stop()
    text = "Turned left {} degrees".format(gyro.angle())
    return text

def turn_right(degrees):
    gyro.reset_angle(0)
    initial_angle = gyro.angle()
    target_angle = initial_angle + degrees
    while gyro.angle() < target_angle:
        robot.drive(0, -150)  # Notice the negative value for turning left
    robot.stop()
    text = "Turned right {} degrees".format(gyro.angle())
    return text




### NEW function to test automation code 

# Function to calculate the centroid of a contour
def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    else:
        return None

# Class to represent a tracked object
class TrackedObject:
    def __init__(self, object_type, contour, centroid):
        self.object_type = object_type
        self.contour = contour
        self.centroid = centroid
        self.last_seen = 0  # Frames since last seen

# Function to detect objects using OpenCV and the camera
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

# Function to move the robot to a specific coordinate (x, y)
def move_to_coordinate(target_x, target_y):
    current_x = 0  # Initial x coordinate
    current_y = 0  # Initial y coordinate
    current_angle = 0  # Initial angle (facing north)

    # Calculate the required distance and angle to the target coordinate
    delta_x = target_x - current_x
    delta_y = target_y - current_y

    distance = math.sqrt(delta_x**2 + delta_y**2)  # Calculate the straight-line distance

    # Calculate the required angle to turn
    target_angle = math.degrees(math.atan2(delta_y, delta_x))
    angle_to_turn = target_angle - current_angle

    # Normalize the angle to the range [-180, 180]
    if angle_to_turn > 180:
        angle_to_turn -= 360
    elif angle_to_turn < -180:
        angle_to_turn += 360

    # Turn the robot to face the target angle
    if angle_to_turn > 0:
        turn_right(angle_to_turn)
    else:
        turn_left(-angle_to_turn)

    # Move the robot forward to the target coordinate
    drive_forward(distance * 10)  # Convert cm to mm

    # Update current position and angle
    current_x = target_x
    current_y = target_y
    current_angle = target_angle

    return f"Moved to coordinate ({target_x}, {target_y})"

# Function to integrate object detection and movement
def detect_and_move_to_object():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the current frame
        detected_objects = detect_objects(frame)

        # If objects are detected, move to the first detected object
        if detected_objects:
            target_object = detected_objects[0]
            target_x, target_y = target_object.centroid
            print(f"Detected object at ({target_x}, {target_y})")

            # Move to the detected object's coordinates
            move_to_coordinate(target_x, target_y)

            # Break after moving to the first detected object
            break

        # Display the frame with detected objects (for debugging purposes)
        for obj in detected_objects:
            cv2.drawContours(frame, [obj.contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, obj.centroid, 5, (0, 0, 255), -1)
            cv2.putText(frame, obj.object_type, obj.centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Test the integrated function
detect_and_move_to_object()