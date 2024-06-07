#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait
from pybricks.robotics import DriveBase
from pybricks.ev3devices import GyroSensor
from pybricks.parameters import Direction
import math


ev3 = EV3Brick()

# Tilslut gyrosensoren til port 1
gyro = GyroSensor(Port.S1)
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

# Constants
WHEEL_DIAMETER = 33.5  # Hjulets størrelse i mm
AXEL_TRACK = 140  # Distancen mellem hjulene

robot = DriveBase(left_motor, right_motor, WHEEL_DIAMETER, AXEL_TRACK)


def drive_forward(distance_mm):
    gyro.reset_angle(0)
    distance_cm = distance_mm * 100

    while robot.distance() <= distance_cm:
        correction = 0 - gyro.angle()
        robot.drive(400, correction)  # 400 mm/seconds

    robot.stop()
    left_motor.brake()
    right_motor.brake()


def drive_backward(distance_mm):
    while robot.distance() > -distance_mm:
        robot.drive(-400, 0)  # 400 mm/seconds

    robot.stop()
    left_motor.brake()
    right_motor.brake()


def turn_right(degrees):
    gyro.reset_angle(0)

    while gyro.angle < degrees:
        robot.drive(400, 10)
    robot.stop()
    left_motor.brake()
    right_motor.brake()
    print(f"Robot turned {gyro.angle()}")


def turn_left(degrees):
    print("test")
