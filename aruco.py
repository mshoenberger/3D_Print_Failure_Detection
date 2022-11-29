#Authors: Michael Shoenberger, Scott Crowner

#Import CV2 and numpy modules for use in our anlaysis
import cv2
import numpy as np


#FUNCTION TO HANDLE DETECTION OF THE ARUCO MARKERS
#RETURNS: Marker ID, rvec_m_c, tm_c
#  PRIORITIZES MARKER 0, ELSE RETURNS MARKER 1
#  IF MARKER 0 OR MARKER 1 ARE NOT IN THERE, IT RETURNS -1, -1, -1 to indicate an image without any 0 or 1 ID aruco markers
def arucoSetup(bgr_img, cube, K):

    #Define the dictionary that will be used for detection
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)  # Get ArUco dictionary for 4x4 with 100 entries

    #Now we can detect the markers and estimate their pose
    corners, ids, _ = cv2.aruco.detectMarkers(image=bgr_img, dictionary=arucoDict)    # Identify markers

    #If the current frame has no aruco codes to detect, return -1 for everything. This will represent our error condition
    if ids is None:
        print("No Aruco found in current frame, returning -1 indicator.")
        return -1, -1, -1

    #If here, we have ID's that are found, let's make sure it contains a 0 or 1 ID. If not, same return
    if 1 not in ids and 0 not in ids:
        print("ID 0 AND 1 NOT FOUND, returning -1 indicator")
        return -1, -1, -1

    #At this point, we know that we have 1 and/or 0 aruco id's, generate the entire set of rvecs and tvecs
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners=corners, markerLength=1.5, cameraMatrix=K, distCoeffs=None)  # Calculate pose of markers
    cv2.aruco.drawDetectedMarkers(image=bgr_img, corners=corners, ids=ids, borderColor=(0, 0, 255)) #And draw the markers on the image

    print(ids)
    #Now we must find the indices of the zero or 1 (prioritize 0 over 1)

    #We will prioritize aruco id 0 as there is no rotational concerns
    if 0 in ids:

        zeroIndex = np.nonzero(ids == 0)[0][0] #Stores the index in format [row] [column], we only need the first element so [0][0]

        #Generate teh proper rvecs and tvecs for the 0 ID marker
        rvec_m_c = rvecs[zeroIndex][0]  # This is a 1x3 rotation vector
        tm_c = tvecs[zeroIndex][0]  # This is a 1x3 translation vector

        return 0, rvec_m_c, tm_c #Return 0 for the id, rvecs_m_c for 0, and tm_c for 0

    else: #Else we know that 1 must be in, and it is the only one of interest that remains. get the index of the ID 1

        oneIndex = np.nonzero(ids == 1)[0][0] #Stores the index in format [row] [column], we only need the first element

        rvec_m_c = rvecs[oneIndex][0]  # This is a 1x3 rotation vector
        tm_c = tvecs[oneIndex][0]  # This is a 1x3 translation vector

        return 1, rvec_m_c, tm_c #Return with 1