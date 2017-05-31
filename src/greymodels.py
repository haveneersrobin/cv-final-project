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

"""

    Grey level models according to:
    
        https://pdfs.semanticscholar.org/5b34/59be44b9eb7d8679ba348db4dfabcd5a8522.pdf
    
    and
    
        Cootes et al. 2000, An Introduction to Active Shape Models

"""

# Load images as grayscales.
def loadImages():
    images = []
    for i in xrange(1,15):
        print "Loading image " + str(i)
        if os.path.isfile(paths.SOBEL+"sobel"+str(i)+".png"):
            sobel = to_grayscale(cv2.imread(paths.SOBEL+"sobel"+str(i)+".png"))
        else:
            sobel = applySobel(load_image(i))
            cv2.imwrite(paths.SOBEL+"sobel"+str(i)+".png",sobel)
        images.append(sobel)
    return np.asarray(images)
    
# Load images.    
def loadImages2():
    images = []
    for i in xrange(1,15):
        images.append(to_grayscale(load_image(i)))
    return np.asarray(images)    
    

# Create the landmark profiles and corresponding covariance matrices.    
def createGreyLevelModel(toothNb, lgth):
    imgs = loadImages2()
    gradimgs = loadImages()
    
    y_streeps = [] 
    vals = np.zeros((40,14,1+lgth*2))
    cov = []
    
    # Load all landmarks of a single tooth, of all persons.
    landmarks_list = load_all_landmarks_for_tooth(toothNb)
    
    # Iterate over all 40 landmarks.
    for lm in range(0,40):
        ys = []
        # Iterate over all persons.
        for person in range(0,14):
                
            # Get the coordinates and values of the points on the normal on the given point lm.
            coords, values = getNormalPoints(landmarks_list[person], lm, lgth, imgs[person])
            _, gradvalues = getNormalPoints(landmarks_list[person], lm, lgth, gradimgs[person])
            values = values.astype(np.float64)
            vals[lm,person] = values
            
            # Calculate the grey level profile for the landmark point and person.
            ys.append(calculateProfile(values, gradvalues))
        
        # Calculate the mean profile of the given point lm.
        y_streeps.append(calculateMeanProfileOfLandmark(ys))
        
        # Calculate covariance of intensities around landmark lm.
        cov.append(calculateCovariance(ys))
        
    return y_streeps, cov

# Calculate covariance matrix of the landmark.        
def calculateCovariance(ys):

    return (np.cov(np.array(ys), rowvar=0))
    
                
# Calculate the profiles of all the given intensities.        
def calculateMeanProfileOfLandmark(intensities):

    return (np.mean(np.array(intensities), axis=0))

    
# Calculate the grey level of the given values.
def calculateProfile(values,gradients):

    sum = np.sum([np.fabs(vals) for vals in values])
    profiles = [g/sum for g in gradients]
    return profiles


# Calculate the derivate grey levels.    
def calculateDerivates(values):

    values_shift = values[1:]
    values = values[:1]
    derivates = values_shift - values
    
    return derivates

    
def main():
    return    

if __name__ == '__main__':
    main()

