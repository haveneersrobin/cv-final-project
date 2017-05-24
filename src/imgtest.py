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
from  scipy.ndimage import filters
from scipy.ndimage import morphology

if __name__ == '__main__':

	radioPath = 'data/Radiographs/'
	segmentPath = 'data/Segmentations/'
	resultPath = 'data/Combinations/'
	clahePath = 'data/CLAHE/'
	landmarkPath = 'data/Landmarks/original/'
	cWhite = (255,255,255)
	cBlue = (255,0,0)
	cGreen = (0,255,0)
    
	if not os.path.isdir(resultPath):
		os.makedirs(resultPath)
	
	if not os.path.isdir(clahePath):
		os.makedirs(clahePath)
            
    #read an image
    
    #for i in xrange(1,15):
    #    imgIndex = "%02d" % i
    #    img1 = cv2.imread(segmentPath+imgIndex+'-0.png')
    #    radioImg = cv2.imread(radioPath+imgIndex+'.tif')
    #    for x in xrange(1,8):
    #        img2 = cv2.imread(segmentPath+imgIndex+'-'+str(x)+'.png')
    #        img1 = cv2.bitwise_or(img1,img2) 
    #    img1[np.where((img1 > [0,0,0]).all(axis = 2))] = [0,255,0] 
    #    cv2.imwrite(resultPath+imgIndex+'.png',img1)
    #    radioImg = cv2.bitwise_or(radioImg,img1)
    #    cv2.imwrite(resultPath+imgIndex+'Overlay.png',radioImg)
    
    
    #visualize landmarks: outline teeth
    
    #for k in xrange(1,2): #15     
    #    imgIndex = "%02d" % k
    #    img = cv2.imread(radioPath+imgIndex+'.tif')
    #    for j in xrange(0,8): #8
    #        data = np.loadtxt(landmarkPath+'landmarks'+str(k)+'-'+str(j+1)+'.txt').astype(int)
    #        data = data.reshape(data.size/2, 2)
    #        dataTuples = [[[data[i,0], data[i,1]]] for i in xrange(len(data))]
    #        dataTuples = np.asarray(dataTuples)
    #        rm = cv2.minAreaRect(dataTuples)
    #        box = cv2.cv.BoxPoints(rm)
    #        box = np.int0(box)
    #        cv2.drawContours(img,[box],0,cBlue,3)            
    #        for i in xrange(len(data)):                
    #            cv2.line(img, (data[i,0],data[i,1]),(data[(i+1) % len(data),0],data[(i+1) % len(data),1]),cWhite,2)        
    #    cv2.imwrite(resultPath+imgIndex+'Outline.png',img)
		
		
	# CLAHE radiographs	
		
	for k in xrange(1,2): #15  -> only do one image for now   
		imgIndex = "%02d" % k
		img = cv2.imread(radioPath+imgIndex+'.tif')
		grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		
		medfilter = filters.median_filter(grayimg,5)
		(h,w) = medfilter.shape
		medfilter_res = cv2.resize(medfilter, (w/3, h/3))
		
		#cv2.imshow('',medfilter_res)
		#cv2.waitKey(0)		
		
		filtered = cv2.bilateralFilter(grayimg, 9, 175, 175)
		
		tophat = morphology.white_tophat(filtered, size=200)
		blackhat = morphology.black_tophat(filtered, size=50)		
		result = cv2.add(filtered,tophat)
		result = cv2.subtract(filtered,blackhat)
		
		result_res = cv2.resize(result, (w/3, h/3)) 
		#cv2.imshow('',result_res)
		#cv2.waitKey(0)
		
		eq = exposure.equalize_adapthist(result,kernel_size=None, clip_limit=0.03, nbins=2**6)
		eq_mul = (eq*255).astype(np.uint8)
		medfilter = filters.median_filter(eq_mul,7)
		eq_res = cv2.resize(medfilter, (w/3, h/3)) 
		#cv2.imshow('',eq_res)
		#cv2.waitKey(0)
		
		medfilter = cv2.GaussianBlur(medfilter,(5,5),0)
		sobelx = cv2.Sobel(medfilter, cv2.CV_64F, 1, 0, ksize=3)
		sobely = cv2.Sobel(medfilter, cv2.CV_64F, 0, 1, ksize=3)
		abs_grad_x = cv2.convertScaleAbs(sobelx)
		abs_grad_y = cv2.convertScaleAbs(sobely)
		gradimage = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
		print gradimage
		gradimage = morphology.grey_closing(gradimage, size=(5,5))
		blur = cv2.GaussianBlur(gradimage,(7,7),0)
		_,bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		print bin
		gradimage = bin.astype(np.uint8)
		print gradimage
		gradimage_res = cv2.resize(gradimage, (w/3, h/3)) 
		cv2.imshow('',gradimage_res)
		cv2.waitKey(0)
		
		'''
		for clip in range(1,11): #0.1 - 1.0
			c = clip/100.
			for bins in range(3,5): # 2^bins (9: 256)
				eq = exposure.equalize_adapthist(grayimg,kernel_size=None, clip_limit=c, nbins=2**bins)
				(h,w) = eq.shape
				eq_res = cv2.resize(eq, (w/3, h/3)) 
				eq_mul = eq_res*255
				result = eq_mul.astype(np.uint8)
				cv2.putText(result, 'CLIP LIMIT=' + str(clip) + ' , NBINS=' + str(2**bins), (10,510), cv2.FONT_HERSHEY_PLAIN,1.5,cv2.cv.CV_RGB(255, 255,255),2)
				cv2.imwrite(clahePath+imgIndex+'clip_'+str(clip)+'_bins_'+str(2**bins)+'.png',result)
		'''
    


             

    
    