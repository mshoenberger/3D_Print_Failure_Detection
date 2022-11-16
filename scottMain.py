# Scott Crowner, Michael Shoenberger
# CSCI 507
# 3D print failure detector

import os
import cv2
import numpy as np
import math
import glob
from printSTL import printSTL

#######################################################################################################################
# DATA AND VARIABLES

IMAGE_DIRECTORY = 'print_failure_detection_images'


#######################################################################################################################
# MAIN CODE
def main():
    ######################################
    # Initial operations

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)  # Get ArUco dictionary

    assert (os.path.exists(IMAGE_DIRECTORY))
    image_file_names = glob.glob(os.path.join(IMAGE_DIRECTORY, "*.jpg"))
    assert (len(image_file_names) > -0)

    bgr_img = cv2.imread(image_file_names[0])
    scale_percent = 25  # percent of original size
    bgr_img = cv2.resize(bgr_img, (int(bgr_img.shape[1] * scale_percent / 100), int(bgr_img.shape[0] * scale_percent / 100)), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)  # make gray image
    #_, thresh_img = cv2.threshold(gray_img, 200, 255, type=cv2.THRESH_BINARY)
    #cv2.imshow("thresholded", thresh_img)
    #cv2.waitKey()
    cv2.imshow("bgr_image", bgr_img)
    cv2.waitKey()

    K = np.array([[2389.0, 0.0, np.shape(bgr_img)[1] / 2],  # Intrinsic camera properties matrix for calculating pose
                  [0.0, 2423.0, np.shape(bgr_img)[0] / 2],
                  [0.0, 0.0, 1.0]])

    t0 = np.array([-4.0, 4.0, 0.0])  # translation vector from marker 0

    cube = np.array([[[0, 0, -1],
                      [0.5, 0.5, 0],
                      [0.5, -0.5, 0],
                      [-0.5, -0.5, 0],
                      [-0.5, 0.5, 0]],
                     [[1, 0, 0],
                      [0.5, 0.5, 0],
                      [0.5, -0.5, 0],
                      [0.5, -0.5, 1],
                      [0.5, 0.5, 1]],
                     [[0, 1, 0],
                      [0.5, 0.5, 0],
                      [-0.5, 0.5, 0],
                      [-0.5, 0.5, 1],
                      [0.5, 0.5, 1]],
                     [[-1, 0, 0],
                      [-0.5, 0.5, 0],
                      [-0.5, -0.5, 0],
                      [-0.5, -0.5, 1],
                      [-0.5, 0.5, 1]],
                     [[0, -1, 0],
                      [0.5, -0.5, 0],
                      [-0.5, -0.5, 0],
                      [-0.5, -0.5, 1],
                      [0.5, -0.5, 1]],
                     [[0, 0, 1],
                      [0.5, 0.5, 1],
                      [0.5, -0.5, 1],
                      [-0.5, -0.5, 1],
                      [-0.5, 0.5, 1]]])

    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(image=bgr_img, dictionary=arucoDict)    # Identify markers

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners=corners, markerLength=1.25, cameraMatrix=K, distCoeffs=None)  # Calculate pose of marker


    rvec_m_c = rvecs[0]  # This is a 1x3 rotation vector
    tm_c = tvecs[0]  # This is a 1x3 translation vector
    cv2.aruco.drawAxis(image=bgr_img, cameraMatrix=K, distCoeffs=None, rvec=rvec_m_c, tvec=tm_c, length=2)  # Draw coordinate axes on marker

    R = cv2.Rodrigues(rvec_m_c)[0]  # Make 3D rotation matrix from calculated rotation vector

    # Build extrinsic camera matrix
    Mext = np.zeros([3, 4])
    Mext[:, 0:3] = R
    Mext[:, 3] = tm_c

    cube2D = np.zeros(np.shape(cube)-np.array([0,0,1]))
    facing = []
    for i in range(np.shape(cube[:,0,0])[0]):
        rot_vec = R @ cube[i,0,:]
        if rot_vec[2] < 0:
            facing.append(True)
        else:
            facing.append(False)
        for j in range(np.shape(cube[0,:,0])[0]):
            moved_cube = cube[i,j,:] + t0
            cube2D[i,j,:] = transform3Dto2D(K, Mext, moved_cube)  # Transform from 3D coordinates to 2D coordinates
    print("original model:", cube2D)

    drawObject(bgr_img, cube2D, facing)  # Draw cube


    # Smooth the image with a Gaussian filter.  If sigma is not provided, it
    # computes it automatically using   sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8.
    gray_img = cv2.GaussianBlur(
        src=gray_img,
        ksize=(0, 0),  # kernel size (should be odd numbers; if 0, compute it from sigma)
        sigmaX=3, sigmaY=3)


    edge_img = gray_img.copy()
    thresh_canny = 10
    # Pick a threshold such that we get a relatively small number of edge points.
    MIN_FRACT_EDGES = 0.05
    MAX_FRACT_EDGES = 0.08
    while np.sum(edge_img) / 255 < MIN_FRACT_EDGES * (gray_img.shape[1] * gray_img.shape[0]):
        print("Decreasing threshold ...")
        thresh_canny *= 0.9
        edge_img = cv2.Canny(
            image=gray_img,
            apertureSize=3,  # size of Sobel operator
            threshold1=thresh_canny,  # lower threshold
            threshold2=3 * thresh_canny,  # upper threshold
            L2gradient=True)  # use more accurate L2 norm
    while np.sum(edge_img) / 255 > MAX_FRACT_EDGES * (gray_img.shape[1] * gray_img.shape[0]):
        print("Increasing threshold ...")
        thresh_canny *= 1.1
        edge_img = cv2.Canny(
            image=gray_img,
            apertureSize=3,  # size of Sobel operator
            threshold1=thresh_canny,  # lower threshold
            threshold2=3 * thresh_canny,  # upper threshold
            L2gradient=True)  # use more accurate L2 norm

    # Run Hough transform.  The output houghLines has size (N,1,4), where N is #lines.
    # The 3rd dimension has the line segment endpoints: x0,y0,x1,y1.
    MIN_HOUGH_VOTES_FRACTION = 0.01
    MIN_LINE_LENGTH_FRACTION = 0.04
    houghLines = cv2.HoughLinesP(
        image=edge_img,
        rho=1,
        theta=math.pi / 180,
        threshold=int(edge_img.shape[1] * MIN_HOUGH_VOTES_FRACTION),
        lines=None,
        minLineLength=int(edge_img.shape[1] * MIN_LINE_LENGTH_FRACTION),
        maxLineGap=10)
    print("Found %d line segments" % len(houghLines))

    # For visualizing the lines, draw on a grayscale version of the image.
    bgr_display = cv2.cvtColor(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    for i in range(0, len(houghLines)):
        l = houghLines[i][0]
        cv2.line(bgr_display, (l[0], l[1]), (l[2], l[3]), (0, 0, 255),
                 thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow("line image", bgr_display)
    cv2.waitKey()



######################################################################################################################
# FUNCTIONS

######################################


if __name__ == "__main__":
    main()
