
import numpy as np
import cv2


#Function to conduct the edge detection process
def edgeDetection(grayBlurredImage):


    #for t in range(30, 150, 20):
    #    low_thresh = t
    #    high_thresh = t
    #    print("low_thresh = %d, high_thresh = %d" % (low_thresh, high_thresh))
    #    edge_img = cv2.Canny(grayBlurredImage, low_thresh, high_thresh, L2gradient=True)
    #    cv2.imshow("Edge image", edge_img)
    #    cv2.waitKey(0)

    for t in range(0, 220, 1):
        for dividor in range(1,5):
            for multiplier in range(1, 3):


                low_thresh = t/dividor
                high_thresh = t * multiplier
                print("low_thresh = %d, high_thresh = %d, divisor = %d, multiplier = %d" % (low_thresh, high_thresh, dividor, multiplier))
                #low_thresh = 100, high_thresh = 100, divisor = 1, multiplier = 1
                edge_img = cv2.Canny(grayBlurredImage, low_thresh, high_thresh, L2gradient=True)
                cv2.imshow("Edge image", edge_img)
                cv2.waitKey(0)

    return 1