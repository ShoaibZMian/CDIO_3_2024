#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait
from pybricks.robotics import DriveBase
import math

ev3 = EV3Brick()


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
    while True:
        robot.straight(-600)
        robot.turn(90)
        robot.straight(-600)
        robot.turn(90)



