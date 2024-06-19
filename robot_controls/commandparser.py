from drivefunc import drive, turn #toggle_rotate
import re

def parse_and_execute(command):
    """
    Parses the given command and executes the corresponding function from drivefunc.py or robotManager.py

    The command format expected is 'actiondistance', e.g., 'forward10', 'backward5', 'turn90', or 'toggle'.

    :param command: The command to parse and execute.
    """
   

    # Split the command into action and distance using regex
    parts = re.match(r'([a-z]+)(-?\d+)', command, re.I)
    if parts:
        items = parts.groups()
    else:
        return "Invalid command or format."

    action = items[0].lower()
    distance = int(items[1])

    if action in ["drive", "turn"]:
       # if action == "toggle":
        #    print("toggle cm")
         #   return toggle_rotate(distance)
        if action == "drive":
            print("Driving {} cm".format(distance))
            return drive(distance)
        elif action == "turn":
            print("Turning {} degrees".format(distance))
            return turn(distance)
    else:
        return "Invalid command or format."
