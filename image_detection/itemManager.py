import math
from collections import defaultdict
 
class ItemManager:
    def __init__(self):
        self.new_state = defaultdict(list)
        self.old_state = defaultdict(list)
        self.all_items_scanned = False
 
    def _calculate_distance(self, x1, y1, x2, y2):
        """Calculates the Euclidean distance between two points."""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
 
    def add_item(self, name, x, y):
        """Adds a new item with its coordinates."""
        self.new_state[name].append((x, y))
 
    def update_closest_ball(self):
        """Updates the state to keep only the closest ball to robot_green."""
        robot_green_coords = self.new_state.get('robot-front')
        if robot_green_coords:
            robot_green_coords = robot_green_coords[0]  # Assuming only one robot_green
            closest_ball = None
            closest_distance = float('inf')
            balls_to_remove = []
 
            for item_name, coords_list in self.new_state.items():
                if item_name.startswith('white-golf-ball'):
                    for coords in coords_list:
                        distance = self._calculate_distance(robot_green_coords[0], robot_green_coords[1], coords[0], coords[1])
                        if distance < closest_distance:
                            if closest_ball is not None:
                                balls_to_remove.append(closest_ball)
                            closest_ball = coords
                            closest_distance = distance
                        else:
                            balls_to_remove.append(coords)
 
            self.new_state['white-golf-ball'] = [closest_ball] if closest_ball else []
            for ball in balls_to_remove:
                if ball in self.new_state['white-golf-ball']:
                    self.new_state['white-golf-ball'].remove(ball)
 
    def get_item(self, name):
        """Retrieves the coordinates of all items by their name."""
        return self.new_state.get(name, None)
    def get_all_items(self):
        """Retrieves all items with their coordinates."""
        if self.all_items_scanned:
            return self.new_state
        else:
            return self.old_state or None
 
    def reset(self):
        """Moves the current state to the old state and resets the new state."""
        self.old_state = self.new_state.copy()
        self.new_state = defaultdict(list)
        self.all_items_scanned = False
 
    def get_prev_state_items(self):
        """Retrieves all items from the old state."""
        return self.old_state
 
    def items_scanned(self):
        """Marks that all items have been scanned."""
        self.all_items_scanned = True
 
# Creating a global instance of ItemManager
item_manager = ItemManager()
 
# Example functions to interact with the global item_manager
def add_item(name, x, y):
    item_manager.add_item(name, x, y)
 
def update_closest_ball():
    item_manager.update_closest_ball()
 
def get_item(name):
    return item_manager.get_item(name)
 
def get_all_items(): 
    return item_manager.get_all_items()
 
def reset():
    item_manager.reset()
 
def get_prev_state_items():
    return item_manager.get_prev_state_items()
 
def items_scanned():
    item_manager.items_scanned()