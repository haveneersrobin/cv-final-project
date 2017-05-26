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
from procustes import *
from landmarks import *
from protocol1 import *
from greymodels import *
from normal import *
from pca import *
from manual_init import *
from radiograph import *

def main():
    # Setup variables.
    tooth_to_fit = 1
    person_to_fit = 1
    
    # Read image and landmarks for tooth.
    landmark_list = load_all_landmarks_for_tooth(tooth_to_fit)
    image = load_image(person_to_fit)
    
    # Build ASM.
    meanShape, ASM_P, eigenvalues = buildASM(landmark_list)
    
    # Manual Init.
    init_tooth = manual_init(meanShape.get_two_lists(), 1, image)
    
    

if __name__ == '__main__':
    main()
