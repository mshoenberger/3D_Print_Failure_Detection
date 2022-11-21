
import numpy as np
import cv2


#Function to conduct the edge detection process
def edgeDetection(grayBlurredImage, modelOutline):

    #Make the outline a flipped color so it is mostly white, will be used to mask the corner edges off
    flippedOutline = cv2.bitwise_not(modelOutline)
    cv2.imshow("flipped", flippedOutline)


    #for t in range(30, 150, 20):
    #    low_thresh = t
    #    high_thresh = t
    #    print("low_thresh = %d, high_thresh = %d" % (low_thresh, high_thresh))
    #    edge_img = cv2.Canny(grayBlurredImage, low_thresh, high_thresh, L2gradient=True)
    #    cv2.imshow("Edge image", edge_img)
    #    cv2.waitKey(0)

    #for t in range(0, 150, 1):
    #    for dividor in range(1,5):
    #        for multiplier in range(1, 3):
    #
    #
    #               low_thresh = t/dividor
    #            high_thresh = t * multiplier
    #            print("low_thresh = %d, high_thresh = %d, divisor = %d, multiplier = %d" % (low_thresh, high_thresh, dividor, multiplier))
    #            #low_thresh = 100, high_thresh = 100, divisor = 1, multiplier = 1
    #            edge_img = cv2.Canny(grayBlurredImage, low_thresh, high_thresh, L2gradient=True)
    #            cv2.imshow("Edge image", edge_img)
    #            cv2.waitKey(0)

    low_thresh = 4
    high_thresh = 25
    edge_img = cv2.Canny(grayBlurredImage, low_thresh, high_thresh, L2gradient=True)

    flippedOutline = cv2.cvtColor(flippedOutline, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.bitwise_and(edge_img, edge_img, mask=flippedOutline)
    cv2.imshow("filtered edge",edge_img)
    cv2.waitKey(0)

    return edge_img

def drawObject(image, model, facing):
    # Connects the points of a face based model, not drawing hidden faces.
    t = 2 # line thickness
    model = model[:,1:,:]
    print("without vector:", model)
    for i in range(np.shape(model[:,0,0])[0]):
        if facing[i] == True:
            for j in range(np.shape(model[0,:,0])[0]-1):
                cv2.line(image, (int(model[i,j,0]), int(model[i,j,1])), (int(model[i,j+1,0]), int(model[i,j+1,1])), (255, 255, 255), t)
            cv2.line(image, (int(model[i,-1,0]), int(model[i,-1,1])), (int(model[i,0,0]), int(model[i,0,1])), (255, 255, 255), t)

#Function to draw the model lines over the edge image. This helps prevent edges from being detected when using the mask


