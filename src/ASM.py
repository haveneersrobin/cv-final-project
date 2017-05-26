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


def buildASM(landmark_list):
    meanShape, result = alignSetOfShapes(landmark_list)
    eigenvalues, ASM_P = pcaManual(result)

    return meanShape, ASM_P, eigenvalues
    
def main():
       

if __name__ == '__main__':
    main()