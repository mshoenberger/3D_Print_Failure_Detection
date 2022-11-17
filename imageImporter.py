

import cv2
import glob
import numpy as np
import os

IMAGE_DIRECTORY = 'print_failure_detection_images' #File path for the images we will use from current file


#Function to bring in file and ensure that error checking is done for improper file paths, etc
def importImageNames():

    #First, ensure that the directory exits or else we we cannot use the images located there
    assert (os.path.exists(IMAGE_DIRECTORY))

    #Now get the image file names in a list, and ensure we have more than 0 images
    image_file_names = glob.glob(os.path.join(IMAGE_DIRECTORY, "*.jpg"))
    assert (len(image_file_names) > -0)

    return image_file_names





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