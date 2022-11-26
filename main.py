#Authors: Michael Shoenberger, Scott Crowner
#Final Project: Print Error Detector

#########################################################################################
#Import the required libraries and other functionality desired

import cv2
import numpy as np
import math

#IMPORT THE FUNCTIONS FROM FILES, USED TO PREVENT GITHUB COLLABORATION ISSUES AND TO PROMOTE SMALL BLOCKS OF FUNCTIONALITY FOR DEBUGGING/EXPANSION
from printSTL import printSTL
from colorIsolation import colorIsolate
from aruco import arucoSetup
from  edgeDetection import edgeDetection
from imageImporter import importImageNames, generateBaseImages
from generateMask import generateMaskedImage
from whiteCount import obtainWhiteCount
from whiteCount import countWhitePixels
from failureAlgorithm import isFailure


#########################################################################################
#GLOBAL VARIABLES, USED TO HAVE A LOCATION TO EASILY MODIFY
# Hard coded a cube model, used to represent an input file with faces defined and a normal vector for each face
h = 1
cube = np.array([[[0, 0, -1],
                  [0.5, 0.5, 0],
                  [0.5, -0.5, 0],
                  [-0.5, -0.5, 0],
                  [-0.5, 0.5, 0]],
                 [[1, 0, 0],
                  [0.5, 0.5, 0],
                  [0.5, -0.5, 0],
                  [0.5, -0.5, h],
                  [0.5, 0.5, h]],
                 [[0, 1, 0],
                  [0.5, 0.5, 0],
                  [-0.5, 0.5, 0],
                  [-0.5, 0.5, h],
                  [0.5, 0.5, h]],
                 [[-1, 0, 0],
                  [-0.5, 0.5, 0],
                  [-0.5, -0.5, 0],
                  [-0.5, -0.5, h],
                  [-0.5, 0.5, h]],
                 [[0, -1, 0],
                  [0.5, -0.5, 0],
                  [-0.5, -0.5, 0],
                  [-0.5, -0.5, h],
                  [0.5, -0.5, h]],
                 [[0, 0, 1],
                  [0.5, 0.5, h],
                  [0.5, -0.5, h],
                  [-0.5, -0.5, h],
                  [-0.5, 0.5, h]]])


#MAIN METHOD

#Main method to conduct the process
def main():

    #Get the image names for a folder as specific in the importImageNames function
    imageNameList = importImageNames()

    #Get the normal image, the gray image, threshold image, and generate a HSV image
    defaultImage = generateBaseImages(imageNameList[11])
    defaultCopy = defaultImage.copy()


    #Get image dimensions, may have value later
    imageHeight = defaultImage.shape[0]
    imageWidth = defaultImage.shape[1]

    #Define camera matrix, was derrived from the camera calibration technqiues/scripts provided in lecture
    K = np.array([[710.0, 0.0, np.shape(defaultImage)[1] / 2],  # Intrinsic camera properties matrix for calculating pose
                  [0.0, 700.0, np.shape(defaultImage)[0] / 2],
                  [0.0, 0.0, 1.0]])


    #Conduct aruco setup and capture the aruco ID of interest, along with the rvec_m_c and tm_c
    markerID, rvec_m_c, tm_c = arucoSetup(defaultImage,cube, K)


    #At this point, we are getting a transition that is not good

    if(markerID == -1):
        print("ERROR: NO MARKER DETECTED. IN A VIDEO THIS WOULD JUST MOVE TO NEXT FRAME OR END THIS ITERATION")


    #Now that we have the aruco ID and the respective marker properties, conduct printSTL to draw the STL object
    userImage, blackMask, outline = printSTL(defaultImage, cube, K, markerID, rvec_m_c, tm_c)


    #Generate the white pixel count of the black mask, so we know how many pixels we have for color isolation
    colorIsolateWhiteCount = countWhitePixels(blackMask)

    #Obtain the number of pixels used actually allowed by the mask, will be used for later error detection algorithm
    maskPixelCount = obtainWhiteCount(blackMask, outline)

    print("number of white Pixels is: " + str(maskPixelCount))


    #At this point we have the current image with the end model drawn on it
    #And at this point we have a black mask that can be used to isolate the rest of the image, lets work on that next

    maskedImage = generateMaskedImage(defaultCopy, blackMask)

    # Now to use a naive method to check if the object is in the bounded area... CHECK FOR COLOR!
    # Isolate the color and see how many pixels there are of that color
    conductColorIsolation = True
    if conductColorIsolation:
        filamentColor = "gray"
        isObjectThere = colorIsolate(maskedImage, filamentColor, colorIsolateWhiteCount)

        if not isObjectThere:
            print("OBJECT NOT IN FRAME VIA COLOR ISOLATION METHOD. PRINT FAILURE DETECTED")
        else:
            print("Object is reasonably intuited to be in frame, continuing program")



    # Make gray blurred image from isolated image for edge detection
    grayMask = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # grayMask = clahe.apply(grayMask[300:360,445:520])
    # cv2.imshow("histequal", grayMask)
    # cv2.waitKey(0)


    #Convert masked image to a gray image
    sigma = 1
    grayMask = cv2.GaussianBlur(
        src=grayMask,
        ksize=(0, 0),  # kernel size (should be odd numbers; if 0, compute it from sigma)
        sigmaX=sigma, sigmaY=sigma)
    cv2.imshow("show gray", grayMask)
    cv2.waitKey(0)


    #Now to generate the edge image
    edge_image = edgeDetection(grayMask, outline, userImage)


    #Now to conduct the analysis of the data and return a failure or not
    if isFailure(edge_image, maskPixelCount):
        print("PRINT FAILURE DETECTED")
    else:
        print("NO FAILURE DETECTED")




#Used to assist in the use of a main function, tells where to point
if __name__ == "__main__":
        main()