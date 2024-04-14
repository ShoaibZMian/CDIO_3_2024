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

def calculate_mapping():
    # measurement is in cm
    point1 = np.array([536, 31])
    point2 = np.array([542, 581])
    distance = 180

    object_width_pixels = np.sqrt(np.power((point1[0]-point2[0]),2) + np.power((point1[1]-point2[1]),2))

    pixel_size_meters = distance / object_width_pixels

    return pixel_size_meters

def calculate_distance(p1, p2, pixel_size_meters):
    distance_in_pixels = np.sqrt(np.power((p1[0]-p2[0]),2) + np.power((p1[1]-p2[1]),2))
    result = distance_in_pixels * pixel_size_meters
    rounded_result = np.round(result,3)
    return rounded_result

ball1 = np.array([154, 349])
ball2 = np.array([522, 300])

print(f"Distance is: {(calculate_distance(ball1, ball2, calculate_mapping()))} cm")
show_img()