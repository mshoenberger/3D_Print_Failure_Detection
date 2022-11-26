
import numpy as np
import cv2
import math

#Function to conduct the edge detection process
def edgeDetection(grayBlurredImage, modelOutline, colorImage):

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
    #Used to make my algorithm logic fit with the drag bar work


    cv2.namedWindow("Edge image")  # Make window
    cv2.createTrackbar("t", "Edge image", 16, 220, nothing)
    cv2.createTrackbar("dividor", "Edge image", 2, 5, nothing)
    cv2.setTrackbarMin("dividor", "Edge image", 1)
    cv2.createTrackbar("multiplier", "Edge image", 2, 3, nothing)
    cv2.setTrackbarMin("multiplier", "Edge image", 1)
    while True:
        t = cv2.getTrackbarPos("t", "Edge image")
        dividor = cv2.getTrackbarPos("dividor", "Edge image")
        multiplier = cv2.getTrackbarPos("multiplier", "Edge image")
        low_thresh = t / dividor
        high_thresh = t * multiplier
        # print("low_thresh = %d, high_thresh = %d, divisor = %d, multiplier = %d" % (low_thresh, high_thresh, dividor, multiplier))
        # low_thresh = 100, high_thresh = 100, divisor = 1, multiplier = 1
        edge_img = cv2.Canny(grayBlurredImage, low_thresh, high_thresh, L2gradient=True)
        cv2.imshow("Edge image", edge_img)
        if not cv2.waitKey(100) == -1:  # On button press:
            break

    # Run Hough transform.  The output houghLines has size (N,1,4), where N is #lines.
    # The 3rd dimension has the line segment endpoints: x0,y0,x1,y1.
    MIN_HOUGH_VOTES_FRACTION = 0.020
    MIN_LINE_LENGTH_FRACTION = 0.0000001
    houghLines = cv2.HoughLinesP(
            image=edge_img,
            rho=1,
            theta=math.pi / 180,
            threshold=int(edge_img.shape[1] * MIN_HOUGH_VOTES_FRACTION),
            lines=None,
            minLineLength=int(edge_img.shape[1] * MIN_LINE_LENGTH_FRACTION),
            maxLineGap=10)

    print("Found %d line segments" % len(houghLines))

    # For visualizing the lines, draw the original image.
    for i in range(0, len(houghLines)):
        l = houghLines[i][0]
        cv2.line(colorImage, (l[0], l[1]), (l[2], l[3]), (0, 0, 255),
                 thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow("edges", colorImage)
    cv2.waitKey(0)


    #Michael's technique to take the edge image, filter out the outline, to get only details on the object faces


    filteredEdge_img = cv2.Canny(grayBlurredImage, low_thresh, high_thresh, L2gradient=True)

    flippedOutline = cv2.cvtColor(flippedOutline, cv2.COLOR_BGR2GRAY)
    filteredEdge_img = cv2.bitwise_and(filteredEdge_img, filteredEdge_img, mask=flippedOutline)
    cv2.imshow("filtered edge",filteredEdge_img)
    cv2.waitKey(0)

    return filteredEdge_img

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


# Nothing function
def nothing(x):
    pass
