import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':

    radioPath = 'Project Data/_Data/Radiographs/'
    segmentPath = 'Project Data/_Data/Segmentations/'
    resultPath = 'Project Data/_Data/Combinations/'
    landmarkPath = 'Project Data/_Data/Landmarks/original/'
    cWhite = (255,255,255)
    cBlue = (255,0,0)
    cGreen = (0,255,0)
    
    if not os.path.isdir(resultPath):
            os.makedirs(resultPath)
            
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
    
    for k in xrange(1,2): #15     
        imgIndex = "%02d" % k
        img = cv2.imread(radioPath+imgIndex+'.tif')
        for j in xrange(0,8): #8
            data = np.loadtxt(landmarkPath+'landmarks'+str(k)+'-'+str(j+1)+'.txt').astype(int)
            data = data.reshape(data.size/2, 2)
            dataTuples = [[[data[i,0], data[i,1]]] for i in xrange(len(data))]
            dataTuples = np.asarray(dataTuples)
            rm = cv2.minAreaRect(dataTuples)
            box = cv2.cv.BoxPoints(rm)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,cBlue,3)            
            for i in xrange(2):#len(data)):                
                cv2.line(img, (data[i,0],data[i,1]),(data[(i+1) % len(data),0],data[(i+1) % len(data),1]),cWhite,2)        
        cv2.imwrite(resultPath+imgIndex+'Outline.png',img)
    
    


             

    
    