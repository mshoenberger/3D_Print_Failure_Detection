# Print STL
# Authors: Michael Shoenberger and Scott Crowner

# File to visually render a 3D model using OpenCV

import numpy as np
import cv2


# Function used to print an opaque rendering of the model or "STL file" given a pose
# INPUTS: original image, model, intrinsic camera matrix K, ArUco ID, ArUco rotation vector, ArUco translation vector
# OUTPUTS: image with model overlayed, line rendering of model, shillouette mask of model, list of model line endpoints
def printSTL(bgr_img, cube, K, id, rvec_m_c, tm_c):
    # Choose translation based on marker
    #MARKER_0_TRANSLATION = np.array([4.0, 4.0, 0.0])  # translation vector from marker 0
    MARKER_0_TRANSLATION = np.array([-3.48, 3.23, 0.0])  # translation vector from marker 0
    MARKER_1_TRANSLATION = np.array([-4.0, -4.0, 0.0])  # translation vector from marker 0
    # Apply translation to pyramid based on which marker is present
    if id == 0:
        t = MARKER_0_TRANSLATION
    if id == 1:  # If it is marker
        t = MARKER_1_TRANSLATION

    R = cv2.Rodrigues(rvec_m_c)[0]  # Make 3D rotation matrix from calculated rotation vector

    # Build extrinsic camera matrix
    Mext = np.zeros([3, 4])
    Mext[:, 0:3] = R
    Mext[:, 3] = tm_c

    # Preallocate spece for 2D line end points and visible face truth list
    cube2D = np.zeros(np.shape(cube)-np.array([0,0,1]))
    facing = []
    # Iterate through model faces
    for i in range(np.shape(cube[:,0,0])[0]):
        rot_vec = R @ cube[i,0,:]   # apply 3D to 3D rotation to face points and face normal vector
        if rot_vec[2] < 0:          # if z component of the face normal vector is negative (facing camera), add True
            facing.append(True)
        else:
            facing.append(False)    # otherwise, False
        # Iterate through face points
        for j in range(np.shape(cube[0,:,0])[0]):
            moved_cube = cube[i,j,:] + t    # apply translation
            cube2D[i,j,:] = transform3Dto2D(K, Mext, moved_cube)  # Transform from 3D coordinates to 2D coordinates

    modelLines = drawObject(bgr_img, cube2D, facing)  # Draw cube, will represent the image that the user will see
    mask, outline = generateBlackImage(bgr_img,cube2D, facing) # Generate a mask for isolating the print, computer will see this

    return bgr_img, mask, outline, modelLines

######################################

# Function to conduct the transformation from 3D to 2D space. Taken from class lecture content
# INPUTS: intrinsic camera matrix K, extrinsic camera matrix M, 3D point to tranform
# OUTPUTS: 2D point
def transform3Dto2D(K, Mext, points):
    # Transforms 3D coordinates to 2D coordinates given intrinsic and extrinsic camera matrices
    # Make point homogenous vector
    location = np.zeros([1, 4])
    location[0, 0:3] = points
    location[0, 3] = 1
    location = np.transpose(location)

    im = K @ Mext @ location  # premultiply to determine 2D points

    p_loc = im / im[2]  # divide by third element b/c homogenous
    p_loc = np.round(p_loc)  # round to whole number for pixels
    x_im = int(p_loc[0])  # convert to integer and store
    y_im = int(p_loc[1])  # convert to integer and store

    return np.array([x_im,y_im])

######################################

# Special rendering function used to draw the object onto the image
# INPUTS: image, 2D model points, visible face truth matrix
# OUTPUTS: 2D model line list, drawing on image
def drawObject(image, model, facing):
    # Connects the points of a face based model, not drawing hidden faces.
    t = 2 # line thickness
    model = model[:,1:,:]
    modelLines = []
    for i in range(np.shape(model[:,0,0])[0]):
        if facing[i] == True:
            for j in range(np.shape(model[0,:,0])[0]-1):
                modelLines.append(np.array([int(model[i,j,0]), int(model[i,j,1]), int(model[i,j+1,0]), int(model[i,j+1,1])]))
                cv2.line(image, (int(model[i,j,0]), int(model[i,j,1])), (int(model[i,j+1,0]), int(model[i,j+1,1])), (255, 255, 255), t)
            modelLines.append(np.array([int(model[i,-1,0]), int(model[i,-1,1]), int(model[i,0,0]), int(model[i,0,1])]))
            cv2.line(image, (int(model[i,-1,0]), int(model[i,-1,1])), (int(model[i,0,0]), int(model[i,0,1])), (255, 255, 255), t)
    return modelLines


# Function to generate a black background with the wireframe image in the location where it should be
# Filled in a solid color, etc
# Modified function that is meant to provide a copy of the hard coded computer model in black and white
# INPUTS: image, 2D model lines, visible lines truth matrix
# OUTPUTS: mask, somethin else
def generateBlackImage(bgr_img, cube2D, facing):
    print("generating black image")

    blackImage = np.zeros(bgr_img.shape, dtype = "uint8") #Generate an all black image

    #Now draw on that image
    drawObject(blackImage, cube2D, facing) #Now black iamge has an outline of the cube

    modelOutline = blackImage.copy() #Create a copy to save

    grayBlackImage = cv2.cvtColor(blackImage, cv2.COLOR_BGR2GRAY) #convert to a gray image

    #conduct a threhsold with OTSU
    thresh, binary_img = cv2.threshold(grayBlackImage, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Find the countours of the model outline (easy)
    cnts, heiarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #and fill the polynomial in a white color, generating the mask
    cv2.fillPoly(blackImage, cnts, [255, 255, 255])

    return blackImage, modelOutline #Return the mask image



