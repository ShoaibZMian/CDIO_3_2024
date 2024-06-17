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
WHEEL_DIAMETER = 55 # Hjulets størrelse i mm
AXEL_TRACK = 171  # Distancen mellem hjulene

robot = DriveBase(left_motor, right_motor, WHEEL_DIAMETER, AXEL_TRACK)

def drive_forward(distance_mm):
    gyro.reset_angle(0)
    robot.reset()
   
    while robot.distance() < distance_mm:
        correction = gyro.angle()  # Negative feedback for correction
        # correction = 0.5 * angle_error  # Apply proportional control to the correction
        # correction = max(min(correction, 30), -30)

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

def turn(degrees):
    # Ensure the degrees are within the range of -365 to 365
    if degrees > 365 or degrees < -365:
        return "Error: Degrees must be between -365 and 365"
    
    gyro.reset_angle(0)
    initial_angle = gyro.angle()
    target_angle = initial_angle + degrees
    
    if degrees > 0:
        while gyro.angle() < target_angle:
            robot.drive(0, -80)  # Turning left
    elif degrees < 0:
        while gyro.angle() > target_angle:
            robot.drive(0, 80)  # Turning right
    
    robot.stop()
    text = "Turned {} degrees".format(degrees)
    return text

