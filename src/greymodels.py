import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import paths
from skimage import exposure, io, img_as_int, img_as_float, img_as_ubyte
from scipy import signal
from scipy.ndimage import filters
from scipy.ndimage import morphology
from procrustes import *
from pca import *
from debug import *
from normal import *

def loadImages():
    images = []
    for i in xrange(0,14):
        imgIndex = "%02d" % (i+1)
        img = cv2.imread(paths.RADIO+imgIndex+'.tif')  
        images.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    return np.asarray(images)
    
    
def loadAllLandmarks():
    
    landmarks = np.zeros((8,14,80)) # for each of the 8 teth, for all 14 mouhts, load all 40 landmarks (80 values)
    landmarksPerToothKind = np.zeros((14,80))
    for i in range(0,8):
        for j in range(0,14):
            landmarksPerToothKind[j] = np.loadtxt(paths.LANDMARK+'landmarks'+str(j+1)+'-'+str(i+1)+'.txt')
        landmarks[i] = landmarksPerToothKind
        
    return landmarks
    
    
def loadAllLandmarksOfTooth(Nb):
    landmarksPerToothKind = np.zeros((14,80))
    for j in range(0,14):
        landmarksPerToothKind[j] = np.loadtxt(paths.LANDMARK+'landmarks'+str(j+1)+'-'+str(Nb)+'.txt')   
    return landmarksPerToothKind
    
def createGreyLevelModel(toothNb, imgs):
    return 
    
    
def getNormals(images, landmarks, length):
    buffers = []
    for idx,img in enumerate(images):
        buffers.append(getNormalPoints(landmarks[idx],length,img))
    buffers = np.asarray(buffers)
    ints = np.asarray([buff[:,2::3] for buff in buffers])
    ints2 = ints[:,0,:]
    print ints2
        
# Calculate the profiles of all the given intensities.        
def calculateAllProfilesOfLandmark(intensities):
    derivs = []
    for idx in range(0,14):
        ints = intensities[idx]
        derivs.append(calculateDerivates(ints))
    y = np.mean(np.asarray(derivs),axis=0)
    return y    
        
# Calculate the grey level of the given values.
def calculateProfile(values):
    derivates = calculateDerivates(values)
    sum = np.sum(np.absolute(derivates))
    y = derivates/sum
    return y


# Calculate the derivate grey levels.    
def calculateDerivates(values):
    values_shift = values[1:]
    values = values[:1]
    derivates = values_shift - values
    return derivates

    
def main():
    print "Loading radiographs.\n"
    imgs = loadImages()
    print "Loading landmarks.\n"
    landmarks = loadAllLandmarksOfTooth(1)
    print "Calculating normals.\n"
    getNormals(imgs, landmarks, 10)
    
    

if __name__ == '__main__':
    main()