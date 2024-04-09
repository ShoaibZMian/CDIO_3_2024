import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

img_path = os.path.join(os.path.dirname(__file__), "pic1.jpg")
print(img_path)
im = cv2.imread(img_path)
im = cv2.resize(im, None, fx=0.7, fy=0.7)
hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# analyzing colors
# h, s, v = cv2.split(hsv_im)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")
# pixel_colors = im.reshape((np.shape(im)[0]*np.shape(im)[1], 3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()
# 
# axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Hue")
# axis.set_ylabel("Saturation")
# axis.set_zlabel("Value")
# plt.show()

light_orange = (1, 70, 70)
dark_orange = (255, 255, 255)
mask = cv2.inRange(hsv_im, light_orange, dark_orange)
result = cv2.bitwise_and(im, im, mask=mask)

rgb_im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
cv2.imshow("image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
