

import cv2
import numpy as np


#Function to conduct color isolation as developed over the weekend, it
def colorIsolate(hsv_image, bgr_image, print_color):

    print("here")

    lowThreshold = 1
    highThreshold = 1

    if print_color == "gray":
        lowThreshold = np.array([8, 0, 143])
        highThreshold = np.array([138, 75, 242])

    #Generate the mask from the threshold values according to the color
    mask = cv2.inRange(hsv_image, lowThreshold, highThreshold)

    #Generate a resulting color image via bitwise_and operation
    result = cv2.bitwise_and(bgr_image,bgr_image, mask= mask)

    cv2.imshow("result", result)
    cv2.waitKey(0)

    #now to conduct the normal process to create a blurred gray image
    gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    sigma = 10
    gray_image = cv2.GaussianBlur(
        src=gray_image,
        ksize=(0, 0),  # kernel size (should be odd numbers; if 0, compute it from sigma)
        sigmaX=sigma , sigmaY=sigma)

    return gray_image