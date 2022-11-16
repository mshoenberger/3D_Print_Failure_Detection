#Authors: Michael Shoenberger, Scott Crowner
#Final Project: Print Error Detector

#########################################################################################
#Import the required libraries and other functionality desired

import os
import cv2
import numpy as np
import math
import glob
from mainMichael import arucoSetup
from printSTL import printSTL

#########################################################################################
#GLOBAL VARAIBLES

IMAGE_DIRECTORY = 'print_failure_detection_images' #File path for the images we will use from current file

h = 0.3
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

def main():

    assert (os.path.exists(IMAGE_DIRECTORY))
    image_file_names = glob.glob(os.path.join(IMAGE_DIRECTORY, "*.jpg"))
    assert (len(image_file_names) > -0)

    bgr_img = cv2.imread(image_file_names[0])
    scale_percent = 25  # percent of original size
    bgr_img = cv2.resize(bgr_img, (int(bgr_img.shape[1] * scale_percent / 100), int(bgr_img.shape[0] * scale_percent / 100)), interpolation=cv2.INTER_AREA)

    K = np.array([[800.0, 0.0, np.shape(bgr_img)[1] / 2],  # Intrinsic camera properties matrix for calculating pose
                  [0.0, 800.0, np.shape(bgr_img)[0] / 2],
                  [0.0, 0.0, 1.0]])

    id, r_vecs, t_vecs = arucoSetup(bgr_img, K)

    printSTL(bgr_img, cube, K, id, r_vecs, t_vecs)

if __name__ == "__main__":
    main()