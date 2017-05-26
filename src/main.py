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
from landmarks import *
from protocol1 import *
from greymodels import *
from normal import *
from pca import *
from manual_init import *
from radiograph import *

def main():
    print "Setup variables."
    # Setup variables.
    tooth_to_fit = 1
    person_to_fit = 1
    maxIter = 100
    
    cWhite = (255,255,255)
    cBlue = (255,0,0)
    cGreen = (0,255,0)    
    
    # Read landmarks for tooth.
    print "Reading landmarks."
    landmark_list = load_all_landmarks_for_tooth(tooth_to_fit)
    
    # Read image.
    print "Reading image."
    image = load_image(person_to_fit)
    
    # Build ASM.
    print "Building ASM model."
    meanShape, ASM_P, eigenvalues, norm = buildASM(landmark_list)
    
    # Manual Init.
    print "Manual initialisation."
    init_tooth = manual_init(meanShape.get_two_lists(), norm, image) 

    print "Scaling radiograph."
    ratio, new_dimensions = scale_radiograph(image, 800)
    image = cv2.resize(image, (new_dimensions[1], new_dimensions[0]), interpolation = cv2.INTER_AREA)      
    
    # Iterate until convergence.
    print "Starting iterations."
    Y, foundPoints = iterate(init_tooth, meanShape, ASM_P, eigenvalues, image, maxIter)
    
    # Plot found points.
    sobelcpy = applySobel(image)
    #sobelcpy = cv2.cvtColor(sobelcpy,cv2.COLOR_GRAY2RGB)
    Nb = len(init_tooth.get_list())/2
    init_toothX, init_toothY = init_tooth.get_two_lists(integer=True)
    foundPointsX, foundPointsY = foundPoints.get_two_lists(integer=True)
    YX, YY = Y.get_two_lists(integer=True)

    for i in range(Nb):
        cv2.line(sobelcpy, (foundPointsX[i],foundPointsY[i]),(foundPointsX[(i+1) % Nb],foundPointsY[(i+1) % Nb]), cWhite, 1)
    for i in range(Nb):
        cv2.line(sobelcpy, (init_toothX[i],init_toothY[i]),(init_toothX[(i+1) % Nb],init_toothY[(i+1) % Nb]), cWhite, 1)   
    # for i in range(Nb):
        # cv2.line(image, (YX[i],YY[i]),(YX[(i+1) % Nb],YY[(i+1) % Nb]), cBlue, 1) 
    cv2.imshow('',sobelcpy)
    cv2.waitKey(0)             
  
if __name__ == '__main__':
    main()
