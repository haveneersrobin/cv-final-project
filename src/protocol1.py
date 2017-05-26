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
from landmarks import *
from ASM import *

# Implementation of Protocol 1 (see paper in Literature).
#   - ASM is a matrix containing eigenvectors. 
#   - meanShape is the mean shape after alignment. This is a Landmarks object.
#   - Y are the next best points to move towards. This is a Landmarks object.
def protocol1(ASM, meanShape, Y):

    #1 - Set b to zeros.
    bPrev = np.ones(ASM.shape[1])
    b = np.zeros(ASM.shape[1])
    theta = 0
    s = 0
    t = np.zeros(2) 
    
    #7 - Check convergence.
    while(np.linalg.norm(bPrev - b) > 1e-7):
    
        # print "Protocol 1 while loop."
        
        #2 - Generate model points.
        # print "Generate model points."
        modelXs = Landmarks(meanShape.get_list() + np.dot(ASM,b))
        
        #3 - Align Y.
        # print "Align model with next best points."
        theta, s, t = alignShapes(Y, modelXs)        
        
        #4 - Project Y.
        # print "Inverse align next best points."
        y = alignFitLandmarks(-theta, 1./s, -t, Y)
        
        #5 - Project y.
        # print "Project next best points in tangent space."
        yProj = Landmarks(y.get_list()/np.dot(y.get_list(),meanShape.get_list()))
        
        #6 - Update b.
        # print "Update b."
        bPrev = b
        b = np.dot(np.transpose(ASM),(yProj.get_list() - meanShape.get_list()))

    # print "END"
    return theta, s, t, b
    
# Apply contraints to b so that
# -3*sqrt(lambda_i) < b_i < 3*sqrt(lamda_i)
# where lambda_i is the i'th eigenvalue.     
def constrainB(b, vals):
    for idx, val in enumerate(vals):
        if b[idx] > 3*np.sqrt(vals[idx]):
            b[idx] = 3*np.sqrt(vals[idx])
        elif b[idx] < -3*np.sqrt(vals[idx]):
            b[idx] = -3*np.sqrt(vals[idx])
    return b

# Improve the current shape until convergence. 
#   - initialPoints is a Landmarks object.
def iterate(initialPoints, meanShape, P, vals, img, maxIter):
    
    # print "Setup iteration variables."
    points = initialPoints
    prevPoints = Landmarks(np.zeros(points.get_list().shape))
    it = 0
    
    while( max(abs(points.get_list()-prevPoints.get_list())) > 1 and it < maxIter ):
        #Increment counter
        it+=1
        print "Iteration ",it
        
        #Find next best points Y, from given points 'points'
        # print "Finding next best points."
        Y = findPoints1(points, 3, img)  
        
        #Find best theta, s, t and b to match Y
        # print "Run protocol 1."
        theta, s, t, b = protocol1(P, meanShape, Y)
        
        # Apply constraints to b
        # print "Apply constraints to b."
        b = constrainB(b, vals)    

        #Find image coordinates of new points
        # print "Determine new tooth points."
        foundPoints = Landmarks(meanShape.get_list() + np.dot(P, b))
        foundPoints = alignFitLandmarks(theta, s, t, foundPoints)
        
        #Store previous points
        prevPoints = points
        
        #Set new points
        points = foundPoints
        
        # print "oldPoints=", prevPoints
        # print "newPoints=", points
        # print "diff=",max(abs(points-prevPoints))
        
    return Y, foundPoints
        
        
# Find best neighbour points according to strongest edge in given gradient image.        
def findPoints1(points, len, img):

    sobel_img = applySobel(img)

    newXs = np.zeros(points.get_list().shape[0]/2)
    newYs = np.zeros(points.get_list().shape[0]/2)
    itbuffer = getNormalPoints(points, len, sobel_img)
    xs = itbuffer[:,0::3]
    ys = itbuffer[:,1::3]
    intensities = itbuffer[:,2::3]
    for idx, ints in enumerate(intensities):
        max_value = np.amax(ints)
        max_index = np.where(ints==max_value)
        indx = np.median(max_index[0]).astype(int)
        newXs[idx] = xs[idx][indx]
        newYs[idx] = ys[idx][indx]
        
    return Landmarks(np.asarray([val for pair in zip(newXs, newYs) for val in pair]))

#Find best neighbour points according to grey level model.    
def findPoints2(points, img):

    return None

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
    grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (h,w,_) = img.shape    
    sobel = cv2.imread('data/Sobel/01SobelGauss.png')   
    sobel = cv2.cvtColor(sobel, cv2.COLOR_RGB2GRAY)    
    maxIter = 100
    mean, Y, foundPoints, initial = iterate(lm, lm[0], sobel, maxIter)
    draw([Y],'blue')
    draw([foundPoints],'red')
    
    initial = np.asarray(initial).astype(int)
    initial = np.reshape(initial, (40, 2), order='C')
    Y = np.asarray(Y).astype(int)
    Y = np.reshape(Y, (40, 2), order='C')
    foundPoints = np.rint(foundPoints).astype(int)
    foundPoints = np.reshape(foundPoints, (40, 2), order='C')
    
    sobelcpy = sobel.copy()
    Nb = len(initial)    
    for i in range(Nb):
        cv2.line(sobelcpy, (foundPoints[i,0],foundPoints[i,1]),(foundPoints[(i+1) % Nb,0],foundPoints[(i+1) % Nb,1]), cGreen, 1)
    for i in range(Nb):
        cv2.line(sobelcpy, (initial[i,0],initial[i,1]),(initial[(i+1) % Nb,0],initial[(i+1) % Nb,1]), cWhite, 1)   
    for i in range(Nb):
        cv2.line(sobelcpy, (Y[i,0],Y[i,1]),(Y[(i+1) % Nb,0],Y[(i+1) % Nb,1]), cBlue, 1)     
    img2 = sobelcpy[650:1100,1200:1500]   
    cv2.imshow('',img2)
    cv2.waitKey(0)  
    cv2.imwrite('data/Sobel/fit2.png',img2)
    cv2.imwrite('data/Sobel/fit.png',sobel[650:1100,1200:1500])
    
if __name__ == '__main__':
    main()