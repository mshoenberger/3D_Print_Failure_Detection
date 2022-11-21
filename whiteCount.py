
import numpy as np
import cv2



#Used to find the count of white pixels when eliminating white pixels that are of the model outline. This will help be more realistic to the analysis done later on
def obtainWhiteCount(blackMask, outline):

    cv2.imshow("og outline", outline)

    returnImage = blackMask.copy()
    flippedOutline = cv2.bitwise_not(outline)

    flippedOutline = cv2.cvtColor(flippedOutline, cv2.COLOR_BGR2GRAY)
    outlinedImage = cv2.bitwise_and(returnImage, returnImage, mask=flippedOutline)

    outlinedImage = cv2.cvtColor(outlinedImage, cv2.COLOR_BGR2GRAY)

    cv2.imshow("OUTLINE ACCOUNT", outlinedImage)


    maskPixelCount = np.sum(outlinedImage == 255)
    return maskPixelCount

