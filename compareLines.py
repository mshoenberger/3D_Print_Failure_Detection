# Compares Hough lines to model lines to determine how many match up

import numpy as np
import cv2

def compareLines(hough, model):
    #print("model:", model)
    #print("hough:", hough)
    mSlope = np.zeros(len(model))
    mIntercept = np.zeros(len(model))
    hSlope = np.zeros(len(hough))
    hIntercept = np.zeros(len(hough))
    for i in range(0,len(model)):
        m = model[i]
        mSlope[i] = (m[3]-m[1])/(m[2]-m[0])
        mIntercept[i] = m[1]-mSlope[i]*m[0]
    for i in range(0,len(hough)):
        h = hough[i]
        h = h[0]
        hSlope[i] = (h[3]-h[1])/(h[2]-h[0])
        hIntercept[i] = h[1]-hSlope[i]*h[0]
    mSlopeTruth = np.zeros(len(mSlope))
    hSlopeTruth = np.zeros(len(hSlope))
    for i in range(0,len(mSlope)):
        for j in range(0,len(hSlope)):
            if abs(mSlope[i] - hSlope[j]) < 0.2:
                mSlopeTruth[i] = True
                hSlopeTruth[j] = True

    #print("mSlopes:", mSlope)
    #print("hSlopes:", hSlope)
    #print("mTruths:", mTruth)
    #print("hTruths:", hTruth)
    nonMatches = np.count_nonzero(mSlopeTruth==0) + np.count_nonzero(hSlopeTruth==0)
    print("number of non-matches:", nonMatches)
    if nonMatches > 5:
        print("PRINT FAILURE DETECTED")