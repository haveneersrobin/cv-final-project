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
from landmarks import *
from radiograph import *

def loadImages():
    images = []
    for i in xrange(1,15):
        images.append(to_grayscale(load_image(i)))
    return np.asarray(images)
    
    
def createGreyLevelModel(toothNb, imgs):
    # itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]
    gs = np.zeros(40,14)
    return 
    for lm in range(0,40):
        for person in range(0,14):
            values = 
            gs[lm,person] = calculateProfile(values)

        
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
    landmarks = load_all_landmarks_for_tooth(1)
    
    

if __name__ == '__main__':
    main()