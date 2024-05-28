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
gyro_sensor = GyroSensor(Port.S1)

left_motor = Motor(Port.B)
right_motor = Motor(Port.C)

# Constants
WHEEL_DIAMETER = 33.5  # Hjulets størrelse i mm
AXEL_TRACK = 140  # Distancen mellem hjulene 

robot = DriveBase(left_motor, right_motor, WHEEL_DIAMETER, AXEL_TRACK)


def drive_forward(distance_mm):
    speed = 400  # Sæt en ønsket hastighed for robotten
    gyro_sensor.reset_angle(0)  # Nulstil gyrosensorens vinkel

    # Beregn rotationen nødvendig for at dække den ønskede afstand
    # Omregningsfaktoren er antallet af grader pr. mm (omkreds af hjulet er pi*diameter)
    degrees_to_rotate = (360 * distance_mm) / (math.pi * WHEEL_DIAMETER)

    # Nulstil motorernes rotationsvinkel
    left_motor.reset_angle(0)
    right_motor.reset_angle(0)

    # Start robotten med at køre lige frem
    robot.drive(speed, 0)

    # Loop indtil den ønskede afstand er tilbagelagt
    while True:
        # Beregn den gennemsnitlige rotationsvinkel for de to motorer
        avg_rotation = (left_motor.angle() + right_motor.angle()) / 2

        # Check om den ønskede afstand er nået
        if abs(avg_rotation) >= degrees_to_rotate:
            break

        # Beregn den aktuelle afvigelse fra den oprindelige retning
        deviation = gyro_sensor.angle()
        correction = deviation * 1.5  # En simpel proportional controller

        # Juster robottens styring baseret på afvigelsen
        robot.drive(speed, -correction)

        wait(10)  # En kort ventetid for at sikre, at løkken ikke kører for hurtigt

    robot.stop()  # Stop robotten, når den ønskede afstand er nået


def drive_backward(distance_mm):
    speed = -400  # Sæt en ønsket hastighed for robotten for at køre bagud
    gyro_sensor.reset_angle(0)  # Nulstil gyrosensorens vinkel

    # Beregn rotationen nødvendig for at dække den ønskede afstand bagud
    # Omregningsfaktoren er antallet af grader pr. mm (omkreds af hjulet er pi*diameter)
    degrees_to_rotate = (360 * distance_mm) / (math.pi * WHEEL_DIAMETER)

    # Nulstil motorernes rotationsvinkel
    left_motor.reset_angle(0)
    right_motor.reset_angle(0)

    # Start robotten med at køre bagud
    robot.drive(speed, 0)  # Den negative hastighed får robotten til at køre bagud

    # Loop indtil den ønskede afstand er tilbagelagt
    while True:
        # Beregn den gennemsnitlige rotationsvinkel for de to motorer
        avg_rotation = (left_motor.angle() + right_motor.angle()) / 2

        # Check om den ønskede afstand er nået
        if abs(avg_rotation) >= degrees_to_rotate:
            break

        # Beregn den aktuelle afvigelse fra den oprindelige retning
        deviation = gyro_sensor.angle()
        correction = deviation * 1.5  # En simpel proportional controller

        # Juster robottens styring baseret på afvigelsen, også mens den kører bagud
        robot.drive(speed, -correction)

        wait(10)  # En kort ventetid for at sikre, at løkken ikke kører for hurtigt

    robot.stop()  # Stop robotten, når den ønskede afstand er nået


def turn_right(degrees):
    initial_angle = gyro_sensor.angle(0)
    target_angle = initial_angle - degrees
    while gyro_sensor.angle() > target_angle:
        robot.drive(-100, 100)
        wait(100)
    robot.stop()

def turn_left(degrees):
    gyro_sensor.reset_angle(0)
    robot.turn(degrees)
    wait(5)
    print(gyro_sensor.angle())
    # initial_angle = gyro_sensor.reset_angle()
   
    # while gyro_sensor.angle() < degrees:
    #     robot.drive(100, -100)
    #     wait(100)
    # robot.stop()
   



