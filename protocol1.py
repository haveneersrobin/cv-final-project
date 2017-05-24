import os
import cv2
import numpy as np
from procrustes import *
from pca import *
from debug import *


def protocol1(ASM, meanShape, Y):
    #1 - Set b to zeros.
    bPrev = np.ones(ASM.shape[1])
    b = np.zeros(ASM.shape[1])
    theta = 0
    s = 0
    t = np.zeros(2)
    print "b", b
    print "bPrev", bPrev    
    
    #7 - Check convergence.
    while(np.linalg.norm(bPrev - b) > 0.00000001):
    
        print "Protocol 1 while loop."
        
        #2 - Generate model points.
        modelXs = meanShape + np.dot(ASM,b)
        # print "ModelXs", modelXs
        
        #3 - Align Y.
        theta, s, t = alignShapes(Y, modelXs)
        print "theta", theta
        print "s", s
        print "t", t
        
        
        #4 - Project Y.
        y = alignFitLandmarks(theta, s, t, Y)
        # print "Y", Y
        # print "y", y
        
        #5 - Project y.
        yProj = y/np.dot(y,meanShape)
        # print "yProj", yProj
        
        #6 - Update b.
        bPrev = b
        b = np.dot(np.transpose(ASM),(yProj - meanShape))
        # print "bPrev", bPrev
        # print "b", b
        
    # print "END"
    return theta, s, t, b
    
def constrainB(b, vals):
    for idx, val in enumerate(vals):
        if b[idx] > 3*np.sqrt(vals[idx]):
            b[idx] = 3*np.sqrt(vals[idx])
        elif b[idx] < -3*np.sqrt(vals[idx]):
            b[idx] = -3*np.sqrt(vals[idx])
    return b
    
def iterate(lm, initialPoints):

    mean, result = alignSetOfShapes(lm)
    vals, P = pcaManual(result)

    nextPoints = findPoints(initialPoints)
    
    Y = nextPoints    
    
    theta, s, t, b = protocol1(P, mean, Y)
    b = constrainB(b, vals)   
    
    foundPoints = mean + np.dot(P, b)
    foundPoints = alignFitLandmarks(theta, s, t, foundPoints)
    return mean, Y, foundPoints
        
        
        
def findPoints(points):
    return None

def main():
    lm = np.zeros((14, 80), dtype=np.float64)
    index = 0
    for file in os.listdir("./data/Landmarks/Original"):
        if file.endswith("1.txt"):
            with open(os.path.join(landmarkPath, file), 'r') as f:
                lm[index] = [line.rstrip('\n') for line in f]
                index += 1
    
    iterate(lm, None)
    
    
if __name__ == '__main__':
    main()
