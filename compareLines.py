#Authors: Michael Shoenberger, Scott Crowner
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
    mSlopeTruth = np.full(len(mSlope), False)
    hSlopeTruth = np.full(len(hSlope), False)
    for i in range(0,len(mSlope)):
        for j in range(0,len(hSlope)):
            if abs(mSlope[i] - hSlope[j]) < 0.2:
                mSlopeTruth[i] = True
                hSlopeTruth[j] = True
    mInterceptTruth = np.full(len(mIntercept), False)
    hInterceptTruth = np.full(len(hIntercept), False)
    for i in range(0, len(mIntercept)):
        for j in range(0, len(hIntercept)):
            if abs(mIntercept[i] - hIntercept[j]) < 20:
                mInterceptTruth[i] = True
                hInterceptTruth[j] = True

    mTruth = mSlopeTruth & mInterceptTruth
    hTruth = hSlopeTruth & hInterceptTruth

    #print("mTruths:", mTruth)
    #print("hTruths:", hTruth)
    nonMatches = np.count_nonzero(mTruth==0) + np.count_nonzero(hTruth==0)
    print("number of non-matches:", nonMatches)
    print("LINE METHOD")
    if nonMatches > 9:
        print("PRINT FAILURE DETECTED")
        failed = True
    else:
        print("NO PRINT FAILURE DETECTED")
        failed = False

    return failed