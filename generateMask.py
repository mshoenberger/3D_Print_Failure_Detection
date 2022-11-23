#Authors: Michael Shoenberger, Scott Crowner



import numpy as np
import cv2

def generateMaskedImage(normalImage, maskImage):


    maskImage = cv2.cvtColor(maskImage, cv2.COLOR_BGR2GRAY)
    filteredImage = cv2.bitwise_and(normalImage, normalImage, mask = maskImage)
    return filteredImage