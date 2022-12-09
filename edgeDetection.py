# Edge detection
# Authors: Michael Shoenberger, Scott Crowner

import numpy as np
import cv2
import math

# Function to conduct the edge detection process
# INPUTS: Gaussian blurred gray isolated image, 2D coordintate list of model, original color image
# OUTPUTS: Canny edge image, Hough lines
def edgeDetection(grayBlurredImage, modelOutline, colorImage):

    #Make the outline a flipped color so it is mostly white, will be used to mask the corner edges off
    flippedOutline = cv2.bitwise_not(modelOutline)
    cv2.imshow("flipped", flippedOutline)

    #Setup window properties and trackbar properties for the canny edge detection process
    cv2.namedWindow("Edge image")  # Make window
    cv2.createTrackbar("t", "Edge image", 33, 220, nothing)
    cv2.createTrackbar("dividor", "Edge image", 1, 5, nothing)
    cv2.setTrackbarMin("dividor", "Edge image", 1)
    cv2.createTrackbar("multiplier", "Edge image", 2, 3, nothing)
    cv2.setTrackbarMin("multiplier", "Edge image", 1)

    #Until the user presses a button to leave the image
    while True:

        #Set a threshold along with a multiplier and divider
        t = cv2.getTrackbarPos("t", "Edge image")
        dividor = cv2.getTrackbarPos("dividor", "Edge image")
        multiplier = cv2.getTrackbarPos("multiplier", "Edge image")
        low_thresh = t / dividor
        high_thresh = t * multiplier

        #condcut the canny process using the slide bar process
        edge_img = cv2.Canny(grayBlurredImage, low_thresh, high_thresh, L2gradient=True)
        cv2.imshow("Edge image", edge_img)
        if not cv2.waitKey(100) == -1:  # On button press:
            break

    # Run Hough transform.  The output houghLines has size (N,1,4), where N is #lines.
    # The 3rd dimension has the line segment endpoints: x0,y0,x1,y1.
    MIN_HOUGH_VOTES_FRACTION = 0.020
    MIN_LINE_LENGTH_FRACTION = 0.0000001

    #Condcut the hough line process to generate a list of lines detected in the image
    houghLines = cv2.HoughLinesP(
            image=edge_img,
            rho=1,
            theta=math.pi / 180,
            threshold=int(edge_img.shape[1] * MIN_HOUGH_VOTES_FRACTION),
            lines=None,
            minLineLength=int(edge_img.shape[1] * MIN_LINE_LENGTH_FRACTION),
            maxLineGap=10)

    #Print out how many line segments were found
    print("Found %d line segments" % len(houghLines))

    # For visualizing the lines, draw the original image.
    for i in range(0, len(houghLines)):
        l = houghLines[i][0]
        cv2.line(colorImage, (l[0], l[1]), (l[2], l[3]), (0, 0, 255),
                 thickness=1, lineType=cv2.LINE_AA)

    #Show the hough lines drawn on the original picture
    cv2.imshow("edges", colorImage)
    cv2.waitKey(0)


    #Michael's technique to take the edge image, filter out the outline, to get only details on the object faces

    #Conduct canny on the same process and save it to a new file
    filteredEdge_img = cv2.Canny(grayBlurredImage, low_thresh, high_thresh, L2gradient=True)

    #Convert color to gray and conduct a mask over it
    flippedOutline = cv2.cvtColor(flippedOutline, cv2.COLOR_BGR2GRAY)
    filteredEdge_img = cv2.bitwise_and(filteredEdge_img, filteredEdge_img, mask=flippedOutline)

    #See the results of our edge mask to get the face image for VOID algorithm
    cv2.imshow("filtered edge",filteredEdge_img)
    cv2.waitKey(0)

    #Return the VOID filtered image, and the hough lines for the linear coorelation algorithm
    return filteredEdge_img, houghLines

# Nothing function
def nothing(x):
    pass
