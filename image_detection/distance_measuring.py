import cv2
import numpy as np
def show_img():
    img_path = "/Users/matt/CDIO_3_2024/image_detection/data/small dataset/train/images/WIN_20240313_09_24_04_Pro_mp4-0002_jpg.rf.fb71d3f1dd2e7327ede6b41f3be4e43a.jpg"
    img = cv2.imread(img_path)

    circles = np.array([
        [536, 31],
        [542, 581]
    ])

    radius = 5

    cv2.circle(img=img, center=circles[0], radius=radius, color=(255,0,0), thickness=-1)
    cv2.putText(img=img, text="text", org=(circles[0,0] - radius, circles[0,1] - radius - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0))

    cv2.imshow("Image with Circles and Labels", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_video(video_path, mp, obj1, obj2):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        one_pixel = calculate_mapping(mp[0],mp[1], 180)

        cv2.putText(frame, f"Fps: {str(int(cap.get(cv2.CAP_PROP_FPS)))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"1 Pixel: {str(np.round(one_pixel, 4))}cm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for point in mp:
            center_x, center_y = point
            cv2.circle(img=frame, center=(center_x, center_y), radius=5, color=(255,0,0), thickness=-1)
        cv2.line(frame, tuple(mp[0]), tuple(mp[1]), (255, 0, 0), thickness=2)
        midpoint = ((mp[0,0] + mp[1,0]) // 2, ((mp[0,1] + mp[1,1]) // 2)-5)
        cv2.putText(frame, "180 cm", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.circle(img=frame, center=(obj1[0], obj1[1]), radius=5, color=(255,0,0), thickness=-1)
        cv2.putText(frame, "Ball 1", obj1-5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.circle(img=frame, center=(obj2[0], obj2[1]), radius=5, color=(255,0,0), thickness=-2)
        cv2.putText(frame, "Ball 1", obj2+5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.line(frame, obj1, obj2, (255, 0, 0), thickness=2)
        midpoint2 = ((obj1[0] + obj2[0]) // 2, ((obj1[1] + obj2[1]) // 2)-5)
        cv2.putText(frame, f"{str(np.round(calculate_distance(obj1, obj2, one_pixel), 4))}cm", midpoint2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        

        cv2.imshow("Object Detection", frame)

        key = cv2.waitKey(1)
        if key == 27:  # 27 = esc
            break

    cap.release()
    cv2.destroyAllWindows()

def calculate_mapping(p1, p2, distance_between):
    object_width_pixels = np.sqrt(np.power((p1[0]-p2[0]),2) + np.power((p1[1]-p2[1]),2))

    pixel_size_meters = distance_between / object_width_pixels

    return pixel_size_meters

def calculate_distance(p1, p2, pixel_size_meters):
    distance_in_pixels = np.sqrt(np.power((p1[0]-p2[0]),2) + np.power((p1[1]-p2[1]),2))
    result = distance_in_pixels * pixel_size_meters
    rounded_result = np.round(result,3)
    return rounded_result

measurement_point1 = np.array([366, 52])
measurement_point2 = np.array([1642, 49])

ball1 = np.array([388, 839])
ball2 = np.array([893, 959])

measurement_points = np.array([
    measurement_point1,
    measurement_point2
])

show_video("/Users/matt/CDIO_3_2024/image_detection/data/test_video.mp4", measurement_points, ball1, ball2)