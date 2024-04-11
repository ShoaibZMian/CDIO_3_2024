import cv2
from cv2.typing import MatLike
import os
import numpy as np

def nothing(x):
    pass

def userIsolateColors(imgs) -> MatLike:
    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin','image',0,255,nothing)
    cv2.createTrackbar('VMin','image',0,255,nothing)
    cv2.createTrackbar('HMax','image',0,179,nothing)
    cv2.createTrackbar('SMax','image',0,255,nothing)
    cv2.createTrackbar('VMax','image',0,255,nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    waitTime = 33
    outputs = imgs

    while(1):
        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])


        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        for i in range(len(imgs)):
            # Create HSV Image and threshold into a range.
            hsv = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            outputs[i] = cv2.bitwise_and(imgs[i],imgs[i], mask= mask)

        horizontal = np.concatenate((outputs[0], outputs[1]), axis=1)
        horizontal2 = np.concatenate((outputs[2], outputs[3]), axis=1)
        total = np.concatenate((horizontal, horizontal2), axis=0)

        # Display output image
        cv2.imshow('image',total)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            return mask
            break

# Load image 
img_names = ["1.jpg", "3.jpg", "6.jpg", "8.jpg"] 
curdir = os.chdir("..\\data")

imgs = [cv2.imread(img_names[0]),
        cv2.imread(img_names[1]),
        cv2.imread(img_names[2]),
        cv2.imread(img_names[3])]

for i in range(len(imgs)):
    imgs[i] = cv2.resize(imgs[i], None, fx=0.4, fy=0.4)


mask = userIsolateColors(imgs)

# cv2.imshow("mask", mask)
# cv2.waitKey(0)
cv2.destroyAllWindows()