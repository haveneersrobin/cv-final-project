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
import greymodels
import greymodels2
from radiograph import *
import paths
from scipy.linalg import hankel

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
            
        #2 - Generate model points.
        modelXs = Landmarks(meanShape.get_list() + np.dot(ASM,b))
        
        #3 - Align Y.
        theta, s, t = alignShapes(Y, modelXs)        
        
        #4 - Project Y.
        y = alignFitLandmarks(-theta, 1./s, -t, Y)
        
        #5 - Project y.
        yProj = Landmarks(y.get_list()/np.dot(y.get_list(),meanShape.get_list()))
        
        #6 - Update b.
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
def iterate(toothNb, initialPoints, meanShape, P, vals, img, maxIter):
    
    Y = points = initialPoints
    prevPoints = Landmarks(np.zeros(points.get_list().shape))
    it = 0
    
    lenProfile = 5
    lenSearch = 7
    profiles, covariances = greymodels33.createGreyLevelModel(toothNb, lenProfile)
     # max(abs(points.get_list()-prevPoints.get_list())) > 1 and 
    middleSum = 0
    # while( middleSum < 16 and it < maxIter ):
    while( max(abs(points.get_list()-prevPoints.get_list())) > 1 and it < maxIter ):
        #Increment counter
        it+=1
        print "Iteration ",it
        # raw_input('Press <ENTER> to continue')
        
        #Find next best points Y, from given points 'points'
        # Y = findPoints1(img, toothNb, points, 5)  
        Y, middleSum = findPoints2(img, toothNb, points, profiles, covariances, lenSearch, lenProfile)
        print middleSum
        # YX, YY = Y.get_two_lists(integer=True)
        # Nb = len(Y.get_list())/2
        # im2 = img.copy()
        # for i in range(Nb):
            # cv2.line(im2, (YX[i],YY[i]),(YX[(i+1) % Nb],YY[(i+1) % Nb]), (255,255,255), 1) 
        # cv2.imshow('',im2)
        # cv2.waitKey(0)          
        
        #Find best theta, s, t and b to match Y
        theta, s, t, b = protocol1(P, meanShape, Y)
        
        # Apply constraints to b
        b = constrainB(b, vals)    

        #Find image coordinates of new points
        foundPoints = Landmarks(meanShape.get_list() + np.dot(P, b))
        foundPoints = alignFitLandmarks(theta, s, t, foundPoints)
        
        #Store previous points
        prevPoints = points
        
        #Set new points
        points = foundPoints
        YX, YY = points.get_two_lists(integer=True)
        Nb = len(points.get_list())/2
        im2 = img.copy()
        for i in range(Nb):
            cv2.line(im2, (YX[i],YY[i]),(YX[(i+1) % Nb],YY[(i+1) % Nb]), (255,0,255), 1) 
        cv2.imshow('',im2)
        cv2.waitKey(0)  
        
    return Y, foundPoints
        
        
# Find best neighbour points according to strongest edge in given gradient image.        
def findPoints1(img, toothNb, points, len):

    sobel_img = applySobel(img)

    newXs = np.zeros(points.get_list().shape[0]/2)
    newYs = np.zeros(points.get_list().shape[0]/2)
    itbuffer = getAllNormalPoints(points, len, sobel_img)
    xs = itbuffer[:,0::3]
    ys = itbuffer[:,1::3]
    intensities = itbuffer[:,2::3]
    # l = np.linspace(-5,5,num=11,dtype=np.uint32)
    # plt.plot(intensities[0])
    # plt.axis(l)
    # plt.show()
    
    for idx, ints in enumerate(intensities):
        max_value = np.amax(ints)
        max_index = np.where(ints==max_value)
        indx = np.median(max_index[0]).astype(int)
        newXs[idx] = xs[idx][indx]
        newYs[idx] = ys[idx][indx]
        
    return Landmarks(np.asarray([val for pair in zip(newXs, newYs) for val in pair]))

#Find best neighbour points according to grey level model.    
def findPoints2(img, toothNb, points, profiles, covariances, lenSearch, lenProfile):
    
    sobel = applySobel(img)
    gray_img = to_grayscale(img)
    
    pX,pY = points.get_two_lists()

    itbuffer = getAllNormalPoints(points, lenSearch, gray_img)
    xs = itbuffer[:,0::3]
    ys = itbuffer[:,1::3]
    intensities = itbuffer[:,2::3]
    
    itbuffer2 = getAllNormalPoints(points, lenSearch, sobel)
    xs = itbuffer2[:,0::3]
    ys = itbuffer2[:,1::3]
    intensities2 = itbuffer2[:,2::3]
    # im2 = img.copy()
    # im2[ys.astype(np.uint),xs.astype(np.uint)] = (255,255,255)
    # cv2.imshow('',im2)
    # cv2.waitKey(0)     
    
    newXs = np.zeros(points.get_list().shape[0]/2)
    newYs = np.zeros(points.get_list().shape[0]/2)
    
    bestFits = np.zeros(2*(lenSearch-lenProfile)+1)
    bestFits2 = []
    
    for idx in range(0,40):
        normalizedSearchProfiles = greymodels33.calculateProfile(intensities[idx],intensities2[idx])   
        subsequences = np.asarray(hankel(normalizedSearchProfiles[:lenProfile*2+1], normalizedSearchProfiles[lenProfile*2:])).T

        #np.asarray(hankel(normalizedSearchProfiles[:lenProfile*2+1], normalizedSearchProfiles[lenProfile*2:])).T voor greymodels2
        #np.asarray(hankel(normalizedSearchProfiles[:lenProfile*2], normalizedSearchProfiles[lenProfile*2-1:])).T voor greymodels
        # print "Search Profile=\n",normalizedSearchProfiles
        bestFit, error = findBestFit(profiles[idx],subsequences, covariances[idx])
        print idx,"Bestfit=",bestFit
        bestFits[bestFit] += 1
        bestFits2.append(bestFit)

    bestFits2 = signal.medfilt(bestFits2,5)
    
    for idx in range(0,40):    
        newidx = np.rint(bestFits2[idx]+lenProfile+1).astype(np.uint32)
        print "newidx=",newidx
        newXs[idx] = xs[idx][newidx]
        newYs[idx] = ys[idx][newidx]

        
    for idx, val in enumerate(bestFits):
        print idx,": ",val
        
    return Landmarks(np.asarray([val for pair in zip(newXs, newYs) for val in pair])), np.sum(bestFits[len(bestFits)/2 - 1 : len(bestFits)/2 + 1])
    
    
# Determine which subsequence is most similar to the profile.    
def findBestFit(profile, subsequences, covariance):

    # print "Model Profile=\n",profile  

    minSeq = -1
    min = np.inf
    for seqNb, sequence in enumerate(subsequences):
        # print "Subsequence=\n",seqNb,sequence
        diff = sequence-profile
        # print "Difference=",diff
        C_inv = np.linalg.inv(covariance)
        f = np.dot(np.dot(diff.T,C_inv),diff)
        # f = np.dot(diff.T, diff)
        # print "f=",f
        (minSeq,min) = (seqNb,f) if f < min else (minSeq,min)
        # raw_input('Press <ENTER> to continue')  

    return minSeq, min
    

def main():
    
    print "Setup variables"
    print "Loading image."
    imgNb = 1
    img = load_image(imgNb)
    toothNb = 1
    print "Loading landmarks."
    points = np.loadtxt(paths.LANDMARK+'landmarks'+str(imgNb)+'-'+str(toothNb)+'.txt').astype(np.int32)
    print "Creating grey level models."
    lenProfile = 3
    lenSearch = 5
    profiles, covariances = createGreyLevelModel(toothNb, lenProfile)    
    print "Calling function."
    lms = findPoints2(img, toothNb, Landmarks(points), profiles, covariances, lenSearch, lenProfile)
    print points
    print lms.get_list()
    
if __name__ == '__main__':
    main()
