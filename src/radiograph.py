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

# Returns the radiograph for the given person
def load_image(person):
    string = "%02d" % (person)
    return cv2.imread(paths.RADIO+string+'.tif')

# Takes an image and a desired height as input.
# Returns:
#   - the ratio by which width and height have to be scaled
#   - the new dimensions to use (h, w)
def scale_radiograph(image, desired_height):
    r = float(desired_height) / image.shape[0]
    dim = (desired_height, int(image.shape[1] * r))

    return r, dim
    
# Convert an RGB image to grayscale.
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
# Apply sobel edge detector.    
def applySobel(img):

    img = processImage(img)
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return sobel
    
# Process the image.    
def processImage(img):

    whiteHatParam = 350
    blackHatParam = 80
    
    img = to_grayscale(img)
    
    img = medianfilter(img,9)
    # img = gaussian(img,5)
    img = cv2.bilateralFilter(img, 9, 150, 150)
    whiteHat = whitehat(img,whiteHatParam)
    blackHat = blackhat(img,blackHatParam)
    img = hatfilter(img, whiteHat, blackHat) 
    img = AHE(img, 0.8, 256)
    # img = cv2.equalizeHist(img)
    
    return img

# Apply median filter.    
def medianfilter(image, size=5):
    return filters.median_filter(image,size)

# Calculate black hat.   
def blackhat(image, size=50):
    return morphology.black_tophat(image, size)

# Calculate white hat.    
def whitehat(image, size=200):
    return morphology.white_tophat(image, size)

# Apply hat filter.    
def hatfilter(image, whiteHat, blackHat):
    tmp = cv2.add(image, whiteHat)
    return cv2.subtract(tmp, blackHat)

# Apply AHE filter.    
def AHE(image, clip, bins):
    tmp = exposure.equalize_adapthist(image,kernel_size=None, clip_limit=clip, nbins=bins)
    return (tmp*255).astype(np.uint8)

# Apply gaussian filter.    
def gaussian(image, kernel=5):
    return cv2.GaussianBlur(image,(kernel,kernel),0)    

def main():
    
    img = load_image(1)
    cv2.imwrite('../report/img/'+'original.png', img)
    gray_img = to_grayscale(img)
    cv2.imwrite('../report/img/'+'grayscale.png', gray_img)    
    median = img = medianfilter(gray_img,9)
    cv2.imwrite('../report/img/'+'median.png', median)    
    bilateral = cv2.bilateralFilter(median, 9, 150, 150)
    cv2.imwrite('../report/img/'+'bilateral.png', bilateral)    
    whiteHat = whitehat(bilateral,350)
    cv2.imwrite('../report/img/'+'whitehat.png', whiteHat)
    blackHat = blackhat(bilateral,80)
    cv2.imwrite('../report/img/'+'blackhat.png', blackHat)
    hatted = hatfilter(bilateral, whiteHat, blackHat) 
    cv2.imwrite('../report/img/'+'hatted.png', hatted)    
    ahe = AHE(img, 0.8, 256)
    cv2.imwrite('../report/img/'+'ahe.png', ahe)    
    sobelx = cv2.Sobel(ahe, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(ahe, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imwrite('../report/img/'+'sobel.png', sobel)
    
if __name__ == '__main__':
    main()
