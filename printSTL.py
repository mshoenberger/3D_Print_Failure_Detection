import numpy as np
import cv2


def printSTL(bgr_img, cube, rvec_m_c, tm_c, R, t):
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
            moved_cube = cube[i,j,:] + t
            cube2D[i,j,:] = transform3Dto2D(K, Mext, moved_cube)  # Transform from 3D coordinates to 2D coordinates
    print("original model:", cube2D)

    drawObject(bgr_img, cube2D, facing)  # Draw cube

    return bgr_img

######################################
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
def drawObject(image, model, facing):
    # Connects the points of a face based model, not drawing hidden faces.
    t = 2 # line thickness
    model = model[:,1:,:]
    print("without vector:", model)
    for i in range(np.shape(model[:,0,0])[0]):
        if facing[i] == True:
            for j in range(np.shape(model[0,:,0])[0]-1):
                cv2.line(image, (int(model[i,j,0]), int(model[i,j,1])), (int(model[i,j+1,0]), int(model[i,j+1,1])), (255, 255, 255), t)
            cv2.line(image, (int(model[i,-1,0]), int(model[i,-1,1])), (int(model[i,0,0]), int(model[i,0,1])), (255, 255, 255), t)
