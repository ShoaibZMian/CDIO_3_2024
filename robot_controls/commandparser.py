from drivefunc import drive, turn, toggle_on, toggle_off
import re

def parse_and_execute(command):
    """
    Parses the given command and executes the corresponding function from drivefunc.py or robotManager.py

    The command format expected is 'actiondistance', e.g., 'forward10', 'backward5', 'turn90'.

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

    if action in ["drive", "turn","on","off"]:
        if action == "drive":
            return drive(distance)
        elif action == "turn":
            return turn(distance)
        if action == "on":
            print("toggle on")
            return toggle_on(distance)
        if action == "off":
            print("toggle off")
            return toggle_off(distance)

    else:
        return "Invalid command or format."