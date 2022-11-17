#Authors: Michael Shoenberger, Scott Crowner
#Final Project: Print Error Detector

#########################################################################################
#Import the required libraries and other functionality desired
import os
import cv2
import numpy as np
import math
import glob

from printSTL import printSTL
from colorIsolation import colorIsolate
from aruco import arucoSetup
from edgeDetection import edgeDetection
from imageImporter import importImageNames, generateBaseImages
from generateMask import generateMaskedImage

#########################################################################################
#GLOBAL VARAIBLES

IMAGE_DIRECTORY = 'print_failure_detection_images' #File path for the images we will use from current file

# Model to be used
h = 0.3     # Height of model
model = np.array([[[0, 0, -1],
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

def main():

    # Get the image names for a folder as specific in the importImageNames function
    imageNameList = importImageNames()

    # Get the normal image, the gray image, threshold image, and generate a HSV image
    defaultImage, grayImage, thresholdImage, hsvImage = generateBaseImages(imageNameList[0])
    defaultCopy = defaultImage.copy()

    # Intrinsic camera properties matrix for calculating pose
    K = np.array([[710.0, 0.0, np.shape(defaultImage)[1] / 2],
                  [0.0, 700.0, np.shape(defaultImage)[0] / 2],
                  [0.0, 0.0, 1.0]])

    # Conduct aruco setup and capture the aruco ID of interest, along with the rvec_m_c and tm_c
    markerID, rvec_m_c, tm_c = arucoSetup(defaultImage, model, K)
    print(markerID)
    print(rvec_m_c)
    print(tm_c)

    # At this point, we are getting a transition that is not good
    if (markerID == -1):
        print("ERROR: NO MARKER DETECTED. IN A VIDEO THIS WOULD JUST MOVE TO NEXT FRAME OR END THIS ITERATION")

    defaultCopy = defaultImage.copy()
    # Now that we have the aruco ID and the respective marker properties, conduct printSTL to draw the STL object
    userImage, blackMask = printSTL(defaultImage, model, K, markerID, rvec_m_c, tm_c)

    cv2.imshow("userImage", userImage)
    cv2.waitKey(0)

    # At this point we have the current image with the end model drawn on it
    # And at this point we have a black mask that can be used to isolate the rest of the image, lets work on that next

    maskedImage = generateMaskedImage(defaultCopy, blackMask)

    # Make gray blurred image from isolated image for edge detection
    grayMask = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #grayMask = clahe.apply(grayMask[300:360,445:520])
    #cv2.imshow("histequal", grayMask)
    #cv2.waitKey(0)
    sigma = 1
    grayMask = cv2.GaussianBlur(
        src=grayMask,
        ksize=(0, 0),  # kernel size (should be odd numbers; if 0, compute it from sigma)
        sigmaX=sigma, sigmaY=sigma)
    cv2.imshow("show gray", grayMask)
    cv2.waitKey(0)

    # Now to generate the edge image
    edge_image = edgeDetection(grayMask, userImage)


if __name__ == "__main__":
    main()