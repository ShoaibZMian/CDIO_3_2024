import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

img_name = "1.jpg"
curdir = os.chdir("..\\data")
img = cv2.imread(img_name)
img = cv2.resize(img, None, fx=0.6, fy=0.6)
hsv_im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# These are wip values, might need tuning or rethink approach
wall_hsv_lower = (0, 143, 169)
wall_hsv_upper = (179, 255, 255)
ball_hsv_lower = (0, 0, 191)
ball_hsv_upper = (179, 120, 255)

wall_mask = cv2.inRange(hsv_im, wall_hsv_lower, wall_hsv_upper)
ball_mask = cv2.inRange(hsv_im, ball_hsv_lower, ball_hsv_upper)

output = cv2.bitwise_and(img, img, mask=wall_mask)
output = cv2.bitwise_or(img, img, mask=ball_mask)

rgb_im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
cv2.imshow("wall mask", wall_mask)
cv2.imshow("ball mask", ball_mask)
cv2.imshow("masked output image", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
