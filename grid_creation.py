import cv2
import numpy as np

img_dir = "/Users/matt/CDIO_3_2024/image_detection/data/small dataset/train/images/WIN_20240313_09_24_04_Pro_mp4-0000_jpg.rf.235c6d77f1f25eadf2b7e3c697c80896.jpg"

img = cv2.imread(img_dir)

corners = np.array([[114, 45], [537, 31], [541, 578],[127, 588]])

for corner in corners:
    cv2.circle(img=img, center=corner, radius=5, color=(255,0,0), thickness=-1)

for x in range(corners[0][0], corners[1][0], 20):
    cv2.line(img, (x, corners[0][1]), (x, corners[3][1]), (0, 255, 0), thickness=1)

for y in range(corners[0][1], corners[3][1], 20):
    cv2.line(img, (corners[0][0], y), (corners[1][0], y), (0, 255, 0), thickness=1)


cv2.line(img, tuple(corners[0]), tuple(corners[1]), (255, 0, 0), thickness=2)
cv2.line(img, tuple(corners[1]), tuple(corners[2]), (255, 0, 0), thickness=2)
cv2.line(img, tuple(corners[2]), tuple(corners[3]), (255, 0, 0), thickness=2)
cv2.line(img, tuple(corners[3]), tuple(corners[0]), (255, 0, 0), thickness=2)

cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()