import os
import cv2 as cv
import numpy as np
from procrustes import *

landmarkPath = 'data/Landmarks/Original'
np.set_printoptions(threshold='nan')

def pcaBuiltIn(landmarks):
    mean, vec =  cv.PCACompute(landmarks, None)
    covar, _ = cv.calcCovarMatrix(landmarks, cv.cv.CV_COVAR_SCRAMBLED | cv.cv.CV_COVAR_SCALE | cv.cv.CV_COVAR_COLS)
    return covar

def pcaManual(landmarks):
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
        i+=1

    nbOfVals = i+1
    print nbOfVals    
    return eigval[:nbOfVals], eigvecs[:,:nbOfVals]
	
def main():
    lm = np.zeros((14, 80), dtype=np.float64)
    index = 0
    for file in os.listdir("./data/Landmarks/Original"):
        if file.endswith("1.txt"):
            with open(os.path.join(landmarkPath, file), 'r') as f:
                lm[index] = [line.rstrip('\n') for line in f]
                index += 1
    mean, result = alignSetOfShapes(lm)
    vals, P = pcaManual(result)
    print vals
    print P.shape


    
if __name__ == '__main__':
    main()
