

import numpy as np
import cv2



def isFailure(edge_image, whiteMaxCount):

    #Generate how many white pixels ARE found
    edgeWhiteCount = np.sum(edge_image == 255)

    print(edgeWhiteCount)
    print(whiteMaxCount)

    totalFractionUsed = (edgeWhiteCount / whiteMaxCount) * 100
    print(totalFractionUsed)

    if totalFractionUsed > 2:
        return True

    return False


