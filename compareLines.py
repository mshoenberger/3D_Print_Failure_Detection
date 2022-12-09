# Linear Correlation
# Authors: Michael Shoenberger, Scott Crowner

import numpy as np
import cv2

# Function that compares Hough lines to model lines to determine how many match up
# INPUTS: Hough line end points array, model line end points array
# OUTPUTS: print failure decision
def compareLines(hough, model):
    # Preallocate space in lists of line slopes and intercepts; m- prefix is for model, h- prefix is for Hough
    mSlope = np.zeros(len(model))
    mIntercept = np.zeros(len(model))
    hSlope = np.zeros(len(hough))
    hIntercept = np.zeros(len(hough))

    # Calcualte slopes and intercepts of model and Hough lines using line equation
    for i in range(0,len(model)):
        m = model[i]
        mSlope[i] = (m[3]-m[1])/(m[2]-m[0])
        mIntercept[i] = m[1]-mSlope[i]*m[0]
    for i in range(0,len(hough)):
        h = hough[i]
        h = h[0]
        hSlope[i] = (h[3]-h[1])/(h[2]-h[0])
        hIntercept[i] = h[1]-hSlope[i]*h[0]

    # Preallocate space in slope truth arrays that determine whether the model and Hough match up
    mSlopeTruth = np.full(len(mSlope), False)
    hSlopeTruth = np.full(len(hSlope), False)
    # Compare each slope. If they match up, put True
    for i in range(0,len(mSlope)):
        for j in range(0,len(hSlope)):
            if abs(mSlope[i] - hSlope[j]) < 0.2:
                mSlopeTruth[i] = True
                hSlopeTruth[j] = True

    # Preallocate space in intercept truth arrays that determine whether the model and Hough match up
    mInterceptTruth = np.full(len(mIntercept), False)
    hInterceptTruth = np.full(len(hIntercept), False)
    # Compare each intercept. If they match up, put True
    for i in range(0, len(mIntercept)):
        for j in range(0, len(hIntercept)):
            if abs(mIntercept[i] - hIntercept[j]) < 20:
                mInterceptTruth[i] = True
                hInterceptTruth[j] = True

    # Check whether both the slopes and intercepts were matched
    mTruth = mSlopeTruth & mInterceptTruth
    hTruth = hSlopeTruth & hInterceptTruth

    # Count how many lines didn't match another line. Unmatched model lines indicate something is missing. Unmatched
    # Hough lines indicate something extra is there. Both indicate failures and add to the failure score.
    nonMatches = np.count_nonzero(mTruth==0) + np.count_nonzero(hTruth==0)
    print("number of non-matches:", nonMatches)
    print("LINE METHOD")

    # A threshold of 9 unmatched lines is used to indicate failure
    if nonMatches > 9:
        print("PRINT FAILURE DETECTED")
        failed = True
    else:
        print("NO PRINT FAILURE DETECTED")
        failed = False

    return failed