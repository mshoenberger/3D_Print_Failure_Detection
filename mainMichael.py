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
    defaultImage, grayImage, thresholdImage, hsvImage = generateBaseImages(imageNameList[1])
    defaultCopy = defaultImage.copy()


    #Get image dimensions, may have value later
    imageHeight = defaultImage.shape[0]
    imageWidth = defaultImage.shape[1]

    #Define camera matrix, was derrived from the camera calibration technqiues/scripts provided in lecture
    K = np.array([[2389.0, 0.0, np.shape(defaultImage)[1] / 2],  # Intrinsic camera properties matrix for calculating pose
                  [0.0, 2423.0, np.shape(defaultImage)[0] / 2],
                  [0.0, 0.0, 1.0]])




    #Conduct aruco setup and capture the aruco ID of interest, along with the rvec_m_c and tm_c
    markerID, rvec_m_c, tm_c = arucoSetup(defaultImage,cube, K)
    print(markerID)
    print(rvec_m_c)
    print(tm_c)

    #At this point, we are getting a transition that is not good

    if(markerID == -1):
        print("ERROR: NO MARKER DETECTED. IN A VIDEO THIS WOULD JUST MOVE TO NEXT FRAME OR END THIS ITERATION")


    defaultCopy = defaultImage.copy()
    #Now that we have the aruco ID and the respective marker properties, conduct printSTL to draw the STL object
    userImage, blackMask = printSTL(defaultImage, cube, K, markerID, rvec_m_c, tm_c)

    cv2.imshow("userImage", userImage)
    cv2.waitKey(0)


    #At this point we have the current image with the end model drawn on it
    #And at this point we have a black mask that can be used to isolate the rest of the image, lets work on that next

    maskedImage = generateMaskedImage(defaultCopy, blackMask)

    #Keep around just in case we want to use color isolation for anything important
    #conductColorIsolation = False

    #if conductColorIsolation:

    #    filamentColor = "gray"
    #    isolatedColorImageBlurry = colorIsolate(hsvImage, defaultImage, filamentColor)
    #    grayImage = isolatedColorImageBlurry



    cv2.imshow("show gray", grayImage)
    cv2.waitKey(0)


    #Now to generate the edge image
    edge_image = edgeDetection(grayImage)


    #cv2.imshow("Test1",grayImage)
    #cv2.imshow("Test",testImage)
    #cv2.imshow("TestEdge", edgeImage)
    #cv2.waitKey(0)


#Used to assist in the use of a main function, tells where to point
if __name__ == "__main__":
        main()