#Authors: Michael Shoenberger and Scott Crowner

import numpy as np
import cv2

#Used to find the count of white pixels when eliminating white pixels that are of the model outline. This will help be more realistic to the analysis done later on
def obtainWhiteCount(blackMask, outline):

    #Copy so we don't modify the OG version
    returnImage = blackMask.copy()

    #use bitwise not to flip the white and black sections, generating the model outline as a mask instead of the environment
    flippedOutline = cv2.bitwise_not(outline)

    #convert the new outline mask to gray
    flippedOutline = cv2.cvtColor(flippedOutline, cv2.COLOR_BGR2GRAY)

    #add the two masks together to genreate a new mask
    outlinedImage = cv2.bitwise_and(returnImage, returnImage, mask=flippedOutline)

    #convert mask to gray image
    outlinedImage = cv2.cvtColor(outlinedImage, cv2.COLOR_BGR2GRAY)

    #count the number of pixels that are whtie
    maskPixelCount = np.sum(outlinedImage == 255)

    #return the masked count
    return maskPixelCount


#return teh number of pixels of a generic image that are wight
def countWhitePixels(image):
    return np.sum(image == 255)

