#Authors: Michael Shoenberger, Scott Crowner

import numpy as np
import cv2

#Simple function to generate a bitwise mask for the VOID algorithm
def generateMaskedImage(normalImage, maskImage):

    #Take the actual mask and the normal image
    maskImage = cv2.cvtColor(maskImage, cv2.COLOR_BGR2GRAY)

    #Generate the bitwise and to isolate the part from the rest of the image
    filteredImage = cv2.bitwise_and(normalImage, normalImage, mask = maskImage)

    #And return teh isolated part
    return filteredImage