

def start_robot():
    print("Robot started")

def receive_image():
    # her modtager vi billeden med cv2
    print("Image received")
   
def shutDownRobot():
    print("robot shutting down")

def fallback():
    print("Error starting over again")


def detect_ball():
    image = receive_image()# vi modtager et billede 
    # image varibale oprettes ved hjælp af recive_image func

    # vi skal bruge image her til at afgøre om ballfound er true eller false

    ball_found = None
    # her ville vi anvende billedgenkendelse til at afgøre, om der er en bold
    if (ball_found == True) :
        print("Detecting ball in the image")  
    else :
        print("No ball found") 
        fallback()
    # returnere true når der er detected en bold på billedet af banen
    return ball_found

def move_to_ball():
    print("Moving to the ball")
    fallback()


def pick_up_ball():
    print("Picking up the ball")
    fallback()

def determine_closest_goal():
    # Vurdering af, hvilken mål der er tættest
    print("Determining closest goal")
    fallback()

def move_to_goal(goal):
    print(f"Moving to the goal at {goal}")
    fallback()

def deliver_ball_to_goal():
    print("Delivering the ball to the goal")
    fallback()

def robot_loop():
    start_robot()
    while detect_ball():
        move_to_ball()
        pick_up_ball()
        closest_goal = determine_closest_goal()
        move_to_goal(closest_goal)
        deliver_ball_to_goal()
        detect_ball()
    else:
        return shutDownRobot()

# Start robottens loop
robot_loop()

