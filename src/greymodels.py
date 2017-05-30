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
import greymodels2
import greymodels33

"""

    Grey level models according to:
    https://pdfs.semanticscholar.org/5b34/59be44b9eb7d8679ba348db4dfabcd5a8522.pdf

"""

# Load images as grayscales.
def loadImages():
    images = []
    for i in xrange(1,15):
        images.append(to_grayscale(load_image(i)))
    return np.asarray(images)
    
def loadImages2():
    images = []
    for i in xrange(1,15):
        print "Loading image " + str(i)
        if os.path.isfile(paths.SOBEL+"sobel"+str(i)+".png"):
            sobel = cv2.imread(paths.SOBEL+"sobel"+str(i)+".png")
        else:
            sobel = applySobel(load_image(i))
            cv2.imwrite(paths.SOBEL+"sobel"+str(i)+".png",sobel)
        images.append(to_grayscale(sobel))
    return np.asarray(images)    
    

# Create the landmark profiles and corresponding covariance matrices.    
def createGreyLevelModel(toothNb, lgth):
    imgs = loadImages()
    ys = np.zeros((40,14,lgth*2))    
    y_streeps = np.zeros((40,lgth*2))    
    vals = np.zeros((40,14,1+lgth*2))
    cov = np.zeros((40,lgth*2,lgth*2))
    
    # Load all landmarks of a single tooth, of all persons.
    landmarks_list = load_all_landmarks_for_tooth(toothNb)
    
    # Iterate over all 40 landmarks.
    for lm in range(0,40):
    
        # Iterate over all persons.
        for person in range(0,14):
                
            # Get the coordinates and values of the points on the normal on the given point lm.
            coords, values = getNormalPoints(landmarks_list[person], lm, lgth, imgs[person])
            values = values.astype(np.float64)
            vals[lm,person] = values
            # print "values=",values
            # print "coordinates=",coords
            
            # Calculate the grey level profile for the landmark point and person.
            ys[lm,person] = calculateProfile(values)
       
        # Calculate the mean profile of the given point lm.
        y_streeps[lm] = calculateMeanProfileOfLandmark(vals[lm])
        
        # Calculate covariance of intensities around landmark lm.
        cov[lm] = calculateCovariance(y_streeps[lm],ys[lm])
    return y_streeps, cov

# Calculate covariance matrix of the landmark.        
def calculateCovariance(y_streep, ys):


    diff = ys-y_streep
    einsum = np.einsum('...i,...j',diff.copy(),diff.copy())
    C = np.mean(einsum, axis=0)
    
    # s = np.zeros((20,20))
    
    # for i in range(0,14):
        # d = np.dot(np.asarray(diff[i]).reshape(20,1),np.asarray(diff[i]).reshape(1,20))
        # print "d=",d.shape
        # s += d
    
    # s = s/14
    
    # print "C=",C
    # print "s=",s
    # print "C-s=",C-s
    
    return C
    
                
# Calculate the profiles of all the given intensities.        
def calculateMeanProfileOfLandmark(intensities):

    derivs = []
    for idx in range(0,14):
        ints = intensities[idx]
        derivs.append(calculateDerivates(ints))
    y = np.mean(np.asarray(derivs),axis=0)
    
    return y    
        
# Calculate the grey level of the given values.
def calculateProfile(values):

    derivates = calculateDerivates(values) 
    # plt.figure(1)
    # plt.subplot(211)
    # plt.title('Values')
    # plt.plot(values)
    
    # plt.subplot(212)
    # plt.title('Derivatives')
    # plt.plot(derivates)
    # plt.show()     
    sum = np.sum(np.absolute(derivates))
    y = derivates/sum
    return y


# Calculate the derivate grey levels.    
def calculateDerivates(values):

    values_shift = values[1:]
    values = values[:-1]
    derivates = values_shift - values
    
    return derivates

    
def main():

    lgth = 10
    toothNb = 1

    profiles, covariances = createGreyLevelModel(toothNb, lgth)
    profiles2, covariances2 = greymodels2.createGreyLevelModel(toothNb, lgth)
    profiles3, covariances3 = greymodels33.createGreyLevelModel(toothNb, lgth)
    
    print profiles[0]
    print profiles2[0]
    print profiles3[0]
    
    print covariances[0][0]
    print covariances2[0][0]
    print covariances3[0][0]
    
    # print profiles.shape
    # print covariances.shape
    

if __name__ == '__main__':
    main()