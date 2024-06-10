from drivefunc import drive_forward, drive_backward, turn_left, turn_right
import re

def parse_and_execute(command):
    """
    Parses the given command and executes the corresponding function from drivefunc.py

    The command format expected is 'actiondistance', e.g., 'forward10' or 'backward5'.

    :param command: The command to parse and execute.
    """
    # Split the command into action and distance using regex
    parts = re.match(r'([a-z]+)(\d+)', command, re.I)
    if parts:
        items = parts.groups()
    else:
        return "Invalid command or format."

    action = items[0]
    distance = int(items[1])

    if action in ["forward", "backward", "left", "right"]:
        if action == "forward":
            print("Driving forward {} cm".format(distance))
            return drive_forward(distance)
        elif action == "backward":
            print("Driving backward {} cm".format(distance))
            return drive_backward(distance)
        elif action == "left":
            print("Turning left {} degrees".format(distance))
            return turn_left(distance)
        elif action == "right":
            print("Turning right {} degrees".format(distance))
            return turn_right(distance)
    else:
        return "Invalid command or format."