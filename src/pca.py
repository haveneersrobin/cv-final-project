import os
import cv2 as cv
import numpy as np
from procrustes import *
from landmarks import *

np.set_printoptions(threshold='nan')

def pcaBuiltIn(landmarks):
    mean, vec =  cv.PCACompute(landmarks, None)
    covar, _ = cv.calcCovarMatrix(landmarks, cv.cv.CV_COVAR_SCRAMBLED | cv.cv.CV_COVAR_SCALE | cv.cv.CV_COVAR_COLS)
    return covar

def pcaManual(landmarks):
    landmarks = [lms.get_list() for lms in landmarks]
    S = np.cov(landmarks, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(-eigvals)
    eigval = eigvals[idx]
    eigvecs = eigvecs[:, idx]   
    totalVar = np.sum(eigval)
    
    i = 0
    var = 0
    while var < 0.99*totalVar:        
        var += eigval[i]
        # print "variance",var/totalVar
        i+=1

    nbOfVals = i
    # print "modes",nbOfVals    
    return eigval[:nbOfVals], eigvecs[:,:nbOfVals]
	
def main():

    lm = load_all_landmarks_for_tooth(1)
    mean, result, _ = alignSetOfShapes(lm)
    vals, P = pcaManual(result)
    
if __name__ == '__main__':
    main()
