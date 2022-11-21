#Authors: Michael Shoenberger, Scott Crowner



import numpy as np
import cv2

def generateMaskedImage(normalImage, maskImage):

    cv2.imshow("Mask image", maskImage)
    cv2.waitKey(0)

    maskImage = cv2.cvtColor(maskImage, cv2.COLOR_BGR2GRAY)
    print(maskImage.shape)

    filteredImage = cv2.bitwise_and(normalImage, normalImage, mask = maskImage)
    cv2.imshow("filtered Image",filteredImage)
    cv2.imshow("normalImage", normalImage)
    cv2.waitKey(0)

    return filteredImage