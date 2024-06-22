import cv2
import numpy as np
import os
import threading
from queue import Queue
from image_recognition import image_recognition_thread
from skimage import transform
from matplotlib import pyplot as plt

shared_list = []
sorted_corners = []
list_lock = threading.Lock()
robot_ready = threading.Condition()
frame_queue = Queue()

# Get path of this file
currentPath = os.path.dirname(os.path.abspath(__file__))

model_path = currentPath + "/models/best_model.pt"
data_yaml_path = currentPath + "/models/data.yaml"
video_path = currentPath + "/models/img01.png"
conf_thresholds = {
    "white-golf-ball": 0.4,
    "robot-front": 0.25,
    "robot-back": 0.25,
    "egg": 0.70,
    "corner1": 0.60,
}


def get_latest_frame():
    try:
        return frame_queue.get()
    except Exception as e:
        return None


def controller():
    image_thread = threading.Thread(
        target=image_recognition_thread,
        args=(
            model_path,
            data_yaml_path,
            video_path,
            conf_thresholds,
            shared_list,
            list_lock,
            robot_ready,
            frame_queue,
        ),
    )

    image_thread.start()

    while True:
        with list_lock:
            if shared_list:
                with robot_ready:
                    robot_ready.notify()

        frame = get_latest_frame()

        if frame is not None:
            sort_corners(shared_list)

            cropped_frame, crop_x_min, crop_y_min = crop_to_corners(
                frame, sorted_corners
            )
            update_corner_coordinates(sorted_corners, crop_x_min, crop_y_min)
            cv2.imshow("cropped_frame", cropped_frame)

            fixed_frame = stretch_corners_to_frame(cropped_frame, sorted_corners)
            cv2.imshow("fixed_frame", fixed_frame)

            # save fixed_frame to models folder
            cv2.imwrite(currentPath + "/models/fixed_frame.png", fixed_frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break


def update_corner_coordinates(corners, crop_x_min, crop_y_min):
    for corner in corners:
        corner.midpoint = (
            corner.midpoint[0] - crop_x_min,
            corner.midpoint[1] - crop_y_min,
        )


def crop_to_corners(frame, detected_objects):
    corners = [obj for obj in detected_objects if obj.name.startswith("corner")]

    if not corners:
        return frame, 0, 0  # If no corners are found, return the original frame

    x_min = min([corner.midpoint[0] for corner in corners])
    y_min = min([corner.midpoint[1] for corner in corners])
    x_max = max([corner.midpoint[0] for corner in corners])
    y_max = max([corner.midpoint[1] for corner in corners])

    cropped_frame = frame[y_min:y_max, x_min:x_max]
    return cropped_frame, x_min, y_min


def stretch_corners_to_frame(cropped_frame, detected_objects):
    # Get cropped_frame dimensions
    orig_height, orig_width = cropped_frame.shape[:2]

    # Find specific corners by name
    corner1 = next(
        (corner for corner in detected_objects if corner.name == "corner1"), None
    )
    corner2 = next(
        (corner for corner in detected_objects if corner.name == "corner2"), None
    )
    corner3 = next(
        (corner for corner in detected_objects if corner.name == "corner3"), None
    )
    corner4 = next(
        (corner for corner in detected_objects if corner.name == "corner4"), None
    )

    if not all([corner1, corner2, corner3, corner4]):
        print("Not all corners detected.")
        return cropped_frame  # Need all 4 corners to perform the transformation

    # Extract corner points
    corners = [
        corner1.midpoint, # type: ignore
        corner2.midpoint, # type: ignore
        corner3.midpoint, # type: ignore
        corner4.midpoint, # type: ignore
    ]

    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = sorted(corners, key=lambda p: (p[1], p[0]))
    if corners[0][0] > corners[1][0]:
        corners[0], corners[1] = corners[1], corners[0]
    if corners[2][0] < corners[3][0]:
        corners[2], corners[3] = corners[3], corners[2]

    # Visualize the detected and ordered corners
    for i, point in enumerate(corners):
        cv2.circle(cropped_frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        cv2.putText(
            cropped_frame,
            f"Corner {i+1}",
            (int(point[0]), int(point[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    # plt.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    # plt.title("Detected and Ordered Corners")
    # plt.show()

    src = np.array(corners, dtype="float32")

    # Define the destination points, the corners of the frame
    dst = np.array(
        [
            [0, 0],
            [orig_width - 1, 0],
            [orig_width - 1, orig_height - 1],
            [0, orig_height - 1],
        ],
        dtype="float32",
    )

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(cropped_frame, M, (orig_width, orig_height))

    return warped


def sort_corners(shared_list):
    sorted_list = []
    sorted_list = sorted(
        [obj for obj in shared_list if obj.name.startswith("corner")],
        key=lambda obj: (obj.midpoint[0], obj.midpoint[1]),
    )
    for I, obj in enumerate(sorted_list, start=1):
        obj.name = f"corner{I}"
        sorted_corners.append(obj)
        print(f"{obj.name}: {obj.midpoint}, {obj.confidence}")


if __name__ == "__main__":
    controller()
