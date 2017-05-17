import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
#from skimage import exposure, io, img_as_int, img_as_float, img_as_ubyte
#from scipy import signal
#from  scipy.ndimage import filters
#from scipy.ndimage import morphology
from ani_filter import anisodiff

# def showImg(image, text=''):
# 	cv2.putText(image, text + ' ' + str(image.shape), (10,1520), cv2.FONT_HERSHEY_PLAIN,3,cv2.cv.CV_RGB(255, 255,255),3)
# 	cv2.imshow('',image)
# 	cv2.waitKey(0)
#
# def medianfilter(image, size=5):
# 	return filters.median_filter(image,size)
#
# def blackhat(image, size=50):
# 	return morphology.black_tophat(image, size)
#
# def whitehat(image, size=200):
# 	return morphology.white_tophat(image, size)
#
# def hatfilter(image, whiteHat, blackHat):
# 	tmp = cv2.add(image, whiteHat)
# 	return cv2.subtract(tmp, blackHat)
#
# def AHE(image, clip, bins):
# 	tmp = exposure.equalize_adapthist(image,kernel_size=None, clip_limit=clip, nbins=bins)
# 	return (tmp*255).astype(np.uint8)
#
# def gaussian(image, kernel=5):
# 	return cv2.GaussianBlur(image,(kernel,kernel),0)

if __name__ == '__main__':

	radioPath = 'Data/Radiographs/'
	segmentPath = 'Project Data/_Data/Segmentations/'
	resultPath = 'Project Data/_Data/Combinations/'
	clahePath = 'Project Data/_Data/CLAHE/'
	hatPath = 'Project Data/_Data/HAT/'
	sobelPath = 'Project Data/_Data/Sobel/'
	landmarkPath = 'Project Data/_Data/Landmarks/original/'
	cWhite = (255,255,255)
	cBlue = (255,0,0)
	cGreen = (0,255,0)
	#
	# if not os.path.isdir(resultPath):
	# 	os.makedirs(resultPath)
	#
	# if not os.path.isdir(clahePath):
	# 	os.makedirs(clahePath)
	#
	# if not os.path.isdir(hatPath):
	# 	os.makedirs(hatPath)
	#
	# if not os.path.isdir(sobelPath):
	# 	os.makedirs(sobelPath)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

	for imgNb in range(1,5):
		imgIndex = "%02d" % imgNb
		print 'Loading image '+imgIndex+' ...'
		img = cv2.imread(radioPath+imgIndex+'.tif')
		print 'Converting to gray scale ...'
		grayimg = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)

		img_blurred = anisodiff(grayimg, 10)

		cv2.namedWindow("main", cv2.WINDOW_AUTOSIZE)
		(width, height) = img_blurred.shape
		img_resize = cv2.resize(img_blurred, (height/2, width/2))
		cv2.imshow("main", img_resize)
		cv2.waitKey(0)

		# w = whitehat(grayimg)
		# b = blackhat(grayimg)
		# hat = hatfilter(grayimg, w, b)

		#plt.hist(grayimg.ravel(),256,[0,256])
		#plt.show()

		#plt.hist(hat.ravel(),256,[0,256])
		#plt.show()

		# print 'Performing hat filter ...'
		# whiteHatParam = 350
		# whiteHat = whitehat(grayimg.copy(),whiteHatParam)
		# blackHatParam = 90
		# blackHat = blackhat(grayimg.copy(),blackHatParam)
		# hatFiltered = hatfilter(grayimg.copy(), whiteHat, blackHat)
		# print 'Equalizing histogram ...'
		# equ = cv2.equalizeHist(hatFiltered)
		# print 'Removing noise ...'
		# gauss = gaussian(equ,7)
		# #write image
		# print 'Writing HAT image to ' + hatPath+imgIndex+'_HAT.png'
		# cv2.imwrite(hatPath+imgIndex+'_HAT.png',equ)
		# print 'Applying Sobel operator ...'
		# gauss = gaussian(hatFiltered,11)
		# sobelx = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=3)
		# sobely = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=3)
		# abs_grad_x = cv2.convertScaleAbs(sobelx)
		# abs_grad_y = cv2.convertScaleAbs(sobely)
		# gradimage = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
		# print 'Writing Sobel image to '+sobelPath+imgIndex+'_Sobel.png'
		# cv2.imwrite(sobelPath+imgIndex+'_Sobel.png',gradimage)
		# print ''
		"""
		closing = cv2.morphologyEx(hatFiltered, cv2.MORPH_CLOSE, kernel)
		cv2.imwrite(hatPath+imgIndex+'_CLOSE.png',closing)
		"""

	"""
	#Show images last
	cv2.namedWindow('',cv.CV_WINDOW_NORMAL)
	showImg(original, 'ORIGINAL')
	#showImg(grayimg, 'GRAYSCALE')
	#showImg(medianFiltered,  'MEDIAN '+str(medianSize))
	#showImg(bilateral, 'BILATERAL')
	#showImg(gauss, 'GAUSSIAN')
	#showImg(AHEFiltered, 'AHEFiltered')
	#showImg(whiteHat, 'WHITEHAT ' + str(whiteHatParam))
	#showImg(blackHat, 'BLACKHAT ' + str(blackHatParam))
	showImg(hatFiltered, 'HATFILTERED')
	"""
