#Authors: Michael Shoenberger, Scott Crowner
#Final Project: Print Error Detector

#########################################################################################
#Import the required libraries and other functionality desired

import os
import cv2
import numpy as np
import math
import glob

#########################################################################################
#GLOBAL VARAIBLES

IMAGE_DIRECTORY = 'print_failure_detection_images' #File path for the images we will use from current file
MARKER_0_TRANSLATION = np.array([4.0, 4.0, 0.0])  # translation vector from marker 0
MARKER_1_TRANSLATION = np.array([-4.0, -4.0, 0.0])  # translation vector from marker 0


#########################################################################################
#NECESSARY FUNCTIONS
#Define functions that can be used for what we want to do throughout the proejct

#Function to conduct aruco code handling from detection, etc
def arucoSetup(bgr_img, K):

    #Define the dictionary that will be used for detection
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)  # Get ArUco dictionary for 4x4 with 100 entries


    #Now we can detect the markers and estimate their pose
    corners, ids, _ = cv2.aruco.detectMarkers(image=bgr_img, dictionary=arucoDict)    # Identify markers
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners=corners, markerLength=1.25, cameraMatrix=K, distCoeffs=None)  # Calculate pose of marker

    return

    for i in range(0, len(rvecs)):
        #Draw their ID's based on how they were detected. This will help us ensure we know they are detected
        cv2.aruco.drawDetectedMarkers(image=bgr_img, corners=corners, ids=ids, borderColor=(0, 0, 255))
        rvec_m_c = rvecs[i]  # This is a 1x3 rotation vector
        tm_c = tvecs[i]  # This is a 1x3 translation vector

        print(rvec_m_c)
        print(tm_c)

        #and draw the pose of them
        cv2.drawFrameAxes(image=bgr_img, cameraMatrix= K, distCoeffs=None, rvec=rvec_m_c, tvec=tm_c, length=1)


    return


def printSTL( ,model):

    #For each of the rvec components
    for i in range(0, len(rvecs)):

        rvec_m_c = rvecs[i]  # This is a 1x3 rotation vector
        tm_c = tvecs[i]  # This is a 1x3 translation vector

        R = cv2.Rodrigues(rvec_m_c)[0]  # Make 3D rotation matrix from calculated rotation vector

        # Build extrinsic camera matrix
        Mext = np.zeros([3, 4])
        Mext[:, 0:3] = R
        Mext[:, 3] = tm_c

        # Apply translation to pyramid based on which marker is present
        if ids[i] == 0:
            moved_cube = model + MARKER_0_TRANSLATION
        if ids[i] == 1: #If it is marker
            moved_cube = model + MARKER_1_TRANSLATION

        cube2D = transform3Dto2D(K, Mext, moved_cube)  # Transform from 3D coordinates to 2D coordinates

        drawCube(bgr_img, cube2D)  # Draw cube

    cv2.imshow("cube",bgr_img)
    cv2.waitKey(0)


#Function to conduct color isolation as developed over the weekend, it
def colorIsolate(hsv_image, bgr_image, print_color):

    print("here")

    lowThreshold = 1
    highThreshold = 1

    if print_color == "gray":
        lowThreshold = np.array([8, 0, 143])
        highThreshold = np.array([138, 75, 242])

    #Generate the mask from the threshold values according to the color
    mask = cv2.inRange(hsv_image, lowThreshold, highThreshold)

    #Generate a resulting color image via bitwise_and operation
    result = cv2.bitwise_and(bgr_image,bgr_image, mask= mask)

    cv2.imshow("result", result)
    cv2.waitKey(0)

    #now to conduct the normal process to create a blurred gray image
    gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    sigma = 10
    gray_image = cv2.GaussianBlur(
        src=gray_image,
        ksize=(0, 0),  # kernel size (should be odd numbers; if 0, compute it from sigma)
        sigmaX=sigma , sigmaY=sigma)

    return gray_image


#Function to draw a Cube by connecting the points provided in the point_list onto the image
def drawCube(image, point_list):
    # Connects the points of a list of 2D coordinates, particularly in a pyramid shape
    t = 2 # line thickness
    cv2.line(image, (int(point_list[0, 0]), int(point_list[0, 1])), (int(point_list[1, 0]), int(point_list[1, 1])), (255, 255, 255), t)
    cv2.line(image, (int(point_list[1, 0]), int(point_list[1, 1])), (int(point_list[2, 0]), int(point_list[2, 1])), (255, 255, 255), t)
    cv2.line(image, (int(point_list[2, 0]), int(point_list[2, 1])), (int(point_list[3, 0]), int(point_list[3, 1])), (255, 255, 255), t)
    cv2.line(image, (int(point_list[3, 0]), int(point_list[3, 1])), (int(point_list[4, 0]), int(point_list[4, 1])), (255, 255, 255), t)
    cv2.line(image, (int(point_list[4, 0]), int(point_list[4, 1])), (int(point_list[0, 0]), int(point_list[0, 1])), (255, 255, 255), t)
    cv2.line(image, (int(point_list[4, 0]), int(point_list[4, 1])), (int(point_list[1, 0]), int(point_list[1, 1])), (255, 255, 255), t)
    cv2.line(image, (int(point_list[4, 0]), int(point_list[4, 1])), (int(point_list[2, 0]), int(point_list[2, 1])), (255, 255, 255), t)
    cv2.line(image, (int(point_list[3, 0]), int(point_list[3, 1])), (int(point_list[0, 0]), int(point_list[0, 1])), (255, 255, 255), t)


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


#Function to handle the processessing of a given image
#First will read and then resize teh image to scale properly
#Will generate an image file, a gray image,
def generateBaseImages(image_name):

    #Read in the current image that we want to analyze
    bgr_img = cv2.imread(image_name)

    #Conduct scaling upon the image to make it smaller
    scale_percent = 100 #25% of OG size
    bgr_img = cv2.resize(bgr_img, (int(bgr_img.shape[1] * scale_percent / 100), int(bgr_img.shape[0] * scale_percent / 100)),
               interpolation=cv2.INTER_AREA)

    #Now we have a rescaled image, time to generate a gray image and threshold image for Aruco codes
    gray_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    sigma = 3
    gray_image = cv2.GaussianBlur(
        src=gray_image,
        ksize=(0, 0),  # kernel size (should be odd numbers; if 0, compute it from sigma)
        sigmaX=sigma , sigmaY=sigma)
    _, thresh_img = cv2.threshold(gray_image, 200, 255, type=cv2.THRESH_BINARY)

    #and finally generate a HSV image in case we ever want to use it
    hsv_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    #Return all 3 generated images as we may need to process them
    return bgr_img, gray_image, thresh_img, hsv_image


#Function to bring in file and ensure that error checking is done for improper file paths, etc
def importImageNames():

    #First, ensure that the directory exits or else we we cannot use the images located there
    assert (os.path.exists(IMAGE_DIRECTORY))

    #Now get the image file names in a list, and ensure we have more than 0 images
    image_file_names = glob.glob(os.path.join(IMAGE_DIRECTORY, "*.jpg"))
    assert (len(image_file_names) > -0)

    return image_file_names

#Function used to transform a 3D object to the 2D plane.
def transform3Dto2D(K, Mext, points):
    # Transforms 3D coordinates to 2D coordinates given intrinsic and extrinsic camera matrices
    p_loc_list = np.zeros([len(points), 2])
    for i in range(0, len(points)):  # iterate through points
        # Make point homogenous vector
        location = np.zeros([1, 4])
        location[0, 0:3] = points[i]
        location[0, 3] = 1
        location = np.transpose(location)

        im = K @ Mext @ location  # premultiply to determine 2D points

        p_loc = im / im[2]  # divide by third element b/c homogenous
        p_loc = np.round(p_loc)  # round to whole number for pixels
        x_im = int(p_loc[0])  # convert to integer and store
        y_im = int(p_loc[1])  # convert to integer and store

        p_loc_list[i, :] = p_loc[0:2, 0]  # drop the points into a list for other uses

    return p_loc_list


#########################################################################################
#MAIN METHOD

#Main method to conduct the process
def main():

    #Get the image names from the currently defined global folder
    imageNameList = importImageNames()

    #Get the normal image, the gray image, threshold image, and teh HSV image
    defaultImage, grayImage, thresholdImage, hsvImage = generateBaseImages(imageNameList[0])


    #Get image dimensions, may have value later
    imageHeight = defaultImage.shape[0]
    imageWidth = defaultImage.shape[1]

    #Define camera matrix, was derrived from the camera calibration technqiues/scripts provided in lecture
    K = np.array([[2389.0, 0.0, np.shape(defaultImage)[1] / 2],  # Intrinsic camera properties matrix for calculating pose
                  [0.0, 2423.0, np.shape(defaultImage)[0] / 2],
                  [0.0, 0.0, 1.0]])

    cube = np.array([[0, 0, 0],  # cube vertex coordinates
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1],
                     [1, 1, 1],
                     [0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0]
                     ])

    print(len(cube))

    for point in cube:
        point = point * 30
        point = point + np.array([50, 50, 0])
        cv2.drawMarker(defaultImage, (point[0], point[1]), (0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=40, thickness=2, line_type=cv2.LINE_AA)


    #Now do aruco detection
    arucoSetup(defaultImage, K, cube)

    cv2.imshow("gray", grayImage)
    cv2.waitKey(0)


    conductColorIsolation = False

    if conductColorIsolation:

        filamentColor = "gray"
        isolatedColorImageBlurry = colorIsolate(hsvImage, defaultImage, filamentColor)
        grayImage = isolatedColorImageBlurry



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