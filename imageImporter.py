# Authors: Michael Shoenberger, Scott Crowner
# File focused on importing images from a folder

import cv2
import glob
import numpy as np
import os

IMAGE_DIRECTORY = 'print_failure_detection_images' #File path for the images we will use from current file

# Function to bring in file and ensure that error checking is done for improper file paths, etc
# INPUTS: nothing
# OUTPUTS: list of image file names including relative location
def importImageNames():

    #First, ensure that the directory exits or else we we cannot use the images located there
    assert (os.path.exists(IMAGE_DIRECTORY))

    #Now get the image file names in a list, and ensure we have more than 0 images
    image_file_names = glob.glob(os.path.join(IMAGE_DIRECTORY, "*.jpg"))
    assert (len(image_file_names) > -0)

    #Return the list of file names
    return image_file_names


# Function to conduct the size thresholding of 25% onto the images. Prevents too large of images
# INPUTS: image name index
# OUTPUTS: image
def generateBaseImages(image_name):

    #Read in the current image that we want to analyze
    bgr_img = cv2.imread(image_name)

    #Conduct scaling upon the image to make it smaller
    scale_percent = 25 #25% of OG size
    bgr_img = cv2.resize(bgr_img, (int(bgr_img.shape[1] * scale_percent / 100), int(bgr_img.shape[0] * scale_percent / 100)),
               interpolation=cv2.INTER_AREA)

    #Return all 3 generated images as we may need to process them
    return bgr_img