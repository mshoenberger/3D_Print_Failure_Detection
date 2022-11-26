

import cv2
import numpy as np


#Function to conduct color isolation as developed over the weekend, it
def colorIsolate(bgr_image, print_color, totalCount):


    #Set the low and high threshold values to a default of 1, will be changed by defined colors
    lowThreshold = 1
    highThreshold = 1

    #If our print will be gray
    if print_color == "gray":

        #Set the thresholds to the values that are best associated with the color gray
        lowThreshold = np.array([8, 0, 143])
        highThreshold = np.array([138, 75, 242])



    #At this point, we have our thresholds set

    #Generate a HSV Image
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    #Generate the mask from the threshold values according to the color
    mask = cv2.inRange(hsv_image, lowThreshold, highThreshold)

    #Generate a resulting color image via bitwise_and operation
    result = cv2.bitwise_and(bgr_image,bgr_image, mask= mask)

    #At this point, we have an image that has isolated the color of the part from the rest of the image. This will occur on the masked image, so this will be used to check for conformity on the piece

    cv2.imshow("result", result)
    cv2.waitKey(0)

    #Now check how many pixels that are NOT black in this image

    numberColoredPixels = np.sum(result != 0)


    fractionalFound = numberColoredPixels / totalCount
    percentFound = fractionalFound * 100

    print("Color Isolation Percent found")
    print(percentFound)

    if percentFound >= 60:
        return True


    return False