import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from skimage import exposure, io, img_as_int, img_as_float, img_as_ubyte
from scipy import signal
from scipy.ndimage import filters
from scipy.ndimage import morphology
from procrustes import *
from pca import *
from debug import *
from normal import *

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
        y = alignFitLandmarks(-theta, 1./s, -t, Y)
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
    
def iterate(lm, initialPoints, img):

    mean, result = alignSetOfShapes(lm)
    vals, P = pcaManual(result)

    nextPoints = findPoints1(initialPoints, 5, img)
    
    Y = nextPoints    
    
    theta, s, t, b = protocol1(P, mean, Y)
    b = constrainB(b, vals)   
    
    foundPoints = mean + np.dot(P, b)
    foundPoints = alignFitLandmarks(theta, s, t, foundPoints)
    return mean, Y, foundPoints
        
        
        
def findPoints1(points, len, img):

    newXs = np.zeros(points.shape[0]/2)
    newYs = np.zeros(points.shape[0]/2)
    zipped = getNormalPoints(points, len, img)
    xs = zipped[:,0::3]
    ys = zipped[:,1::3]
    intensities = zipped[:,2::3]
    for idx, ints in enumerate(intensities):
        max_value = np.amax(ints)
        max_index = np.where(ints==max_value)
        indx = np.median(max_index[0]).astype(int)
        newXs[idx] = xs[idx][indx]
        newYs[idx] = ys[idx][indx]
        
    zipped = [val for pair in zip(newXs, newYs) for val in pair]
    return zipped
    

def main():

    cWhite = (255,255,255)
    cBlue = (255,0,0)
    cGreen = (0,255,0)

    lm = np.zeros((14, 80), dtype=np.float64)
    index = 0
    for file in os.listdir("./data/Landmarks/Original"):
        if file.endswith("1.txt"):
            with open(os.path.join(landmarkPath, file), 'r') as f:
                lm[index] = [line.rstrip('\n') for line in f]
                index += 1
    img = cv2.imread('data/Radiographs/01.tif')            
    sobel = cv2.imread('data/Sobel/01SobelGauss.png')    
    mean, Y, foundPoints = iterate(lm, lm[1], sobel)
    print foundPoints
    print Y
    draw([foundPoints],'red')
    draw([Y],'blue')
    
    foundPoints = np.reshape(foundPoints, (2, 40), order='F')
    
    for i in range(len(foundPoints)):
        cv2.line(img, (foundPoints[i,0],foundPoints[i,1]),(foundPoints[(i+1) % len(foundPoints),0],foundPoints[(i+1) % len(foundPoints),1]), cWhite, 2)
    img2 = cv2.resize(img, (w/3, h/3))
    cv2.imshow('',img2)
    cv2.waitKey(0)    
if __name__ == '__main__':
    main()
