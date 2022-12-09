# VOID failure determination
# Authors: Michael Shoenberger, Scott Crowner

import numpy as np
import cv2

# Function to help with detection of if a failure occured in the VOID algorithm
# INPUTS: canny edge image, number of white pixels in canny edge image
# OUTPUTS: print failure decision
def isFailure(edge_image, whiteMaxCount):

    #Generate how many white pixels ARE found
    edgeWhiteCount = np.sum(edge_image == 255)

    #Generate the fraction of white pixels determined out of the total white pixels that could exist
    totalFractionUsed = (edgeWhiteCount / whiteMaxCount) * 100
    print(totalFractionUsed) #print the percentage

    #If we are above the 6% threshold
    if totalFractionUsed > 6:
        return True #Return true

    #Else return false
    return False


