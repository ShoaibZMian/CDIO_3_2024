

def start_robot():
    print("Robot started")

def receive_image():
    # her modtager vi billeden med cv2
    print("Image received")
   

def detect_ball(image):
    # Denne funktion ville anvende billedgenkendelse til at afgøre, om der er en bold
    print("Detecting ball in the image")
    # returnere true når der er detected en bold på billedet af banen
    return True

def move_to_ball():
    print("Moving to the ball")

def pick_up_ball():
    print("Picking up the ball")

def determine_closest_goal():
    # Vurdering af, hvilken mål der er tættest
    print("Determining closest goal")


def move_to_goal(goal):
    print(f"Moving to the goal at {goal}")

def deliver_ball_to_goal():
    print("Delivering the ball to the goal")

def robot_loop():
    start_robot()
    while True:
        image = receive_image()
        if detect_ball(image):
            move_to_ball()
            pick_up_ball()
            closest_goal = determine_closest_goal()
            move_to_goal(closest_goal)
            deliver_ball_to_goal()
            continue
        else:
            print("No ball detected, shutting down.")
            return False

# Start robottens loop
robot_loop()

