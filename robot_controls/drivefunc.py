from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait, StopWatch
from pybricks.robotics import DriveBase


import math

ev3 = EV3Brick()

left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
toggle_motor = Motor(Port.A)

# Constants
WHEEL_DIAMETER = 55  # Hjulets størrelse i mm
AXEL_TRACK = 172  # Distancen mellem hjulene
DRIVE_SPEED = 500  # Speed for driving forward/backward
TURN_SPEED = 200  # Speed for turning
WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER

robot = DriveBase(left_motor, right_motor, WHEEL_DIAMETER, AXEL_TRACK)

def drive_forward(distance_mm):
    # Calculate the number of motor degrees needed to drive the specified distance
    rotations = distance_mm / WHEEL_CIRCUMFERENCE
    degrees = rotations * 360

    left_motor.reset_angle(0)
    right_motor.reset_angle(0)

    left_motor.run_target(DRIVE_SPEED, degrees, wait=False)
    right_motor.run_target(DRIVE_SPEED, degrees)

    left_motor.brake()
    right_motor.brake()

    text = "Driven forward {} mm".format(distance_mm)
    return text

def drive_backward(distance_mm):
    # Calculate the number of motor degrees needed to drive the specified distance
    rotations = distance_mm / WHEEL_CIRCUMFERENCE
    degrees = rotations * 360

    left_motor.reset_angle(0)
    right_motor.reset_angle(0)

    left_motor.run_target(-DRIVE_SPEED, -degrees, wait=False)
    right_motor.run_target(-DRIVE_SPEED, -degrees)

    left_motor.brake()
    right_motor.brake()

    text = "Driven backward {} mm".format(distance_mm)
    return text

def turn(degrees):
    # Ensure the degrees are within the range of -365 to 365
    if degrees > 365 or degrees < -365:
        return "Error: Degrees must be between -365 and 365"

    # Calculate the number of motor degrees needed to turn the specified degrees
    turn_circumference = math.pi * AXEL_TRACK
    rotations = (degrees / 360) * turn_circumference / WHEEL_CIRCUMFERENCE
    motor_degrees = rotations * 360

    left_motor.reset_angle(0)
    right_motor.reset_angle(0)

    if degrees > 0:
        left_motor.run_target(TURN_SPEED, motor_degrees, wait=False)
        right_motor.run_target(TURN_SPEED, -motor_degrees)
    elif degrees < 0:
        left_motor.run_target(TURN_SPEED, -motor_degrees, wait=False)
        right_motor.run_target(TURN_SPEED, motor_degrees)

    left_motor.brake()
    right_motor.brake()

    text = "Turned {} degrees".format(degrees)
    return text

def toggle_rotate(distance):
    # Rotate the toggle motor 360 degrees
    print("før motorstart")
    toggle_motor.run_target(500, 360)
    print("inde efter motor run target")
    
    text = "Toggle motor rotated 360 degrees"
    return text

# Example usage in the command parser
