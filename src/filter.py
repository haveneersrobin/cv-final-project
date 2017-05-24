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
from  scipy.ndimage import filters
from scipy.ndimage import morphology

def showImg(image, text=''):
    cv2.putText(image, text + ' ' + str(image.shape), (10,1520), cv2.FONT_HERSHEY_PLAIN,3,cv2.cv.CV_RGB(255, 255,255),3)
    cv2.imshow('',image)
    cv2.waitKey(0)

def medianfilter(image, size=5):
    return filters.median_filter(image,size)

def blackhat(image, size=50):
    return morphology.black_tophat(image, size)

def whitehat(image, size=200):
    return morphology.white_tophat(image, size)

def hatfilter(image, whiteHat, blackHat):
    tmp = cv2.add(image, whiteHat)
    return cv2.subtract(tmp, blackHat)

def AHE(image, clip, bins):
    tmp = exposure.equalize_adapthist(image,kernel_size=None, clip_limit=clip, nbins=bins)
    return (tmp*255).astype(np.uint8)

def gaussian(image, kernel=5):
    return cv2.GaussianBlur(image,(kernel,kernel),0)

def sobelWrite(img, name):
    print 'Applying Sobel operator ...'
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    gradimage = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    print 'Writing Sobel image to '+paths.SOBEL+name+'.png'
    cv2.imwrite(paths.SOBEL+name+'.png',gradimage)

if __name__ == '__main__':

    cWhite = (255,255,255)
    cBlue = (255,0,0)
    cGreen = (0,255,0)

    if not os.path.isdir(paths.RESULT):
        os.makedirs(paths.RESULT)

    if not os.path.isdir(paths.CLAHE):
        os.makedirs(paths.CLAHE)

    if not os.path.isdir(paths.HAT):
        os.makedirs(paths.HAT)

    if not os.path.isdir(paths.SOBEL):
        os.makedirs(paths.SOBEL)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    for imgNb in range(10,11):
        imgIndex = "%02d" % imgNb
        print 'Loading image '+imgIndex+' ...'
        img = cv2.imread(paths.RADIO+imgIndex+'.tif')
        print 'Converting to gray scale ...'
        grayimg = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
        print np.mean(grayimg)

        """
        sobelWrite(grayimg,imgIndex+'Sobel10')

        print 'Removing noise ...'
        median = medianfilter(grayimg,9)
        (h,w) = median.shape
        img2 = cv2.resize(median,(w/3,h/3))
        cv2.imshow('',img2)
        cv2.waitKey(0)

        # plt.hist(grayimg.ravel(),256,[0,256])
        # plt.show()

        # plt.hist(hat.ravel(),256,[0,256])
        # plt.show()

        print 'Performing hat filter ...'
        whiteHatParam = 350
        whiteHat = whitehat(median.copy(),whiteHatParam)
        blackHatParam = 90
        blackHat = blackhat(median.copy(),blackHatParam)
        hatFiltered = hatfilter(median.copy(), whiteHat, blackHat)
        print 'Equalizing histogram ...'
        equ = cv2.equalizeHist(hatFiltered)
        #write image
        print 'Applying Sobel operator ...'
        gauss = gaussian(equ,11)

        sobelWrite(gauss, imgIndex+'SobelGauss10')
        """
