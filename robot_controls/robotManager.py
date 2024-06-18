import sys
import os
import math


# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from drivefunc import drive_forward, turn_left, turn_right

# Function to convert coordinates to distance and angle
def calculate_distance_and_angle(target_x, target_y, current_x=0, current_y=0):
    delta_x = target_x - current_x
    delta_y = target_y - current_y

    distance = math.sqrt(delta_x**2 + delta_y**2)
    target_angle = math.degrees(math.atan2(delta_y, delta_x))

    return distance, target_angle

# global variable to store robot state
robot_moving = False

# Get all detected items
def robot_process_items(items):

    # Check if robot is moving
    if robot_moving:
        return "Robot is currently moving"

    # Move the robot to the ball
    move_to_ball(items)


# Function to move the robot to the ball
def move_to_ball(items):
    robot_moving = True
    
    robot_green = items['robot-front']
    robot_red = items['robot-back']
    ball = items['white-golf-ball']

    # Debugging: Print coordinates
    print("Ball coordinates:", ball)
    print("Robot green coordinates:", robot_green)

    # Calculate the angle between the robot's front and back
    robot_angle = calculate_angle(robot_green[0], robot_red[0])
    print(f"Robot angle: {robot_angle} degrees")

    ball_x, ball_y = ball[0]
    robot_x, robot_y = robot_green[0]

    # Calculate the distance and angle
    distance, angle_to_ball = calculate_distance_and_angle(ball_x, ball_y, robot_x, robot_y)
    print(f"Distance to ball: {distance} units, Angle to ball: {angle_to_ball} degrees")

    # Adjust the angle to the ball based on the robot's angle
    adjusted_angle = angle_to_ball - robot_angle
    print(f"Adjusted angle: {adjusted_angle} degrees")

    # Convert distance to mm
    distance_mm = distance * 3.7  # Assuming 1 unit = 37 mm, adjust if necessary

    # Normalize the adjusted angle to the range [-180, 180]
    if adjusted_angle > 180:
        adjusted_angle -= 360
    elif adjusted_angle < -180:
        adjusted_angle += 360

    # Move the robot
    #if adjusted_angle > 0:
    #    turn_right(adjusted_angle)
    #else:
    #    turn_left(-adjusted_angle)
    
    #drive_forward(distance_mm)

    robot_moving = False

def calculate_angle(robot_front, robot_back):
    dx = robot_front[0] - robot_back[0]
    dy = robot_front[1] - robot_back[1]
    angle = math.atan2(dy, dx)
    
    # Convert the angle from radians to degrees
    angle = math.degrees(angle)
    
    return angle