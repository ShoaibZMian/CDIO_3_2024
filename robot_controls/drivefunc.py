#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait
from pybricks.robotics import DriveBase
from pybricks.ev3devices import GyroSensor
import math

ev3 = EV3Brick()
# Tilslut gyrosensoren til port 4
#gyro_sensor = GyroSensor(Port.D)

left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

# Constants
WHEEL_DIAMETER = 33.5  # Hjulets størrelse i mm
AXEL_TRACK = 140  # Distancen mellem hjulene 

robot = DriveBase(left_motor, right_motor, WHEEL_DIAMETER, AXEL_TRACK)


def drive_forward(distance_mm):
    robot.straight(distance_mm)

def drive_backward(distance_cm):
    print("dummy")

def turn_left(degrees):
    robot.turn(degrees)

def turn_right(degrees):
    print("test")
#     initial_angle = gyro_sensor.angle()
#     while gyro_sensor.angle() - initial_angle < degrees:
#         left_motor.run(100)
#         right_motor.run(-100)
#     left_motor.stop()
#     right_motor.stop()



