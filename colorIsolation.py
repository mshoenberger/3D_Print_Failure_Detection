# Color isolation
# Authors: Michael Shoenberger, Scott Crowner

import cv2
import numpy as np

# Function to conduct color isolation - part of VOID
# INPUTS: image, color of filament, number of pixels in mask
# OUTPUTS: decision on whether print has detatched from the print bed
def colorIsolate(bgr_image, print_color, totalCount):


    #Set the low and high threshold values to a default of 1, will be changed by defined colors
    lowThreshold = 1
    highThreshold = 1

    #If our print will be gray
    if print_color == "gray":

        #Set the thresholds to the values that are best associated with the color gray
        lowThreshold = np.array([8, 0, 143])
        highThreshold = np.array([138, 75, 242])

    elif print_color == "dark gray":

        #Set the thresholds to the values that are best associated with the color gray
        lowThreshold = np.array([84, 0, 59])
        highThreshold = np.array([138, 75, 242])

    elif print_color == "white":
        #Set the thresholds to the values that are best associated with the color white
        #lowThreshold = np.array([55, 0, 163])
        #highThreshold = np.array([110, 33, 242])

        lowThreshold = np.array([6, 27, 118])
        highThreshold = np.array([29, 121, 255])

    elif print_color == "red":
        #Set the thresholds to the values that are best associated with the color red
        lowThreshold = np.array([104, 118, 81])
        highThreshold = np.array([179, 255, 255])

    elif print_color == "blue":
        #Set the thresholds to the values that are best associated with the color blue
        lowThreshold = np.array([29, 52, 188])
        highThreshold = np.array([136, 255, 255])

    elif print_color == "green":
        #Set the thresholds to the values that are best associated with the color green
        lowThreshold = np.array([34, 78, 88])
        highThreshold = np.array([80, 255, 255])

    else:
        print("UNDEFINED PRINT COLOR. RETURNING")
        return -1



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

    #Calculate the fraction as a percentage
    fractionalFound = numberColoredPixels / totalCount
    percentFound = fractionalFound * 100

    #Print the percentage found
    print("Color Isolation Percent found")
    print(percentFound)

    #Set teh 60% Threshold vlaue that we want, return true if above
    if percentFound >= 60:
        return True

    #else we return false
    return False