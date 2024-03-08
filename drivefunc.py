#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait
import math

ev3 = EV3Brick()


left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

# Constants
WHEEL_DIAMETER = 2  # ændre til realistisk hjuldiameter
WHEEL_CIRCUMFERENCE = WHEEL_DIAMETER * math.pi
GEAR_RATIO = 1  # ændre til realistisk gear ratio

# Function to calculate the motor rotation in degrees based on the distance to travel
def calculate_rotation_from_distance(distance_cm, gear_ratio, wheel_circumference):
    # Calculate the number of wheel rotations needed
    wheel_rotations = distance_cm / wheel_circumference
    motor_rotations = wheel_rotations * gear_ratio
    motor_degrees = motor_rotations * 360
    return motor_degrees


def drive_forward(distance_cm):
    rotation = calculate_rotation_from_distance(distance_cm, GEAR_RATIO, WHEEL_CIRCUMFERENCE)
    left_motor.run_angle(500, rotation, wait=False)
    right_motor.run_angle(500, rotation, wait=True) 

def drive_backward(distance_cm):
    rotation = calculate_rotation_from_distance(distance_cm, GEAR_RATIO, WHEEL_CIRCUMFERENCE)
    left_motor.run_angle(500, -rotation, wait=False)
    right_motor.run_angle(500, -rotation, wait=True) 

def turn_left(degrees):
    rotation = degrees * 3.5  # "placeholder" Simplified conversion, adjust based on robot's turning radius
    left_motor.run_target(500, -rotation, wait=False) 
    right_motor.run_target(500, rotation, wait=True) 

def turn_right(degrees):
    rotation = degrees * 3.5 
    left_motor.run_target(500, rotation, wait=False)
    right_motor.run_target(500, -rotation, wait=True)



