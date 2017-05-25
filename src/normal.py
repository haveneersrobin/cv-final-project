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

def createLineIterator(P1, P2, img):
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer

    
def getNormalPoints(points, lgth, grayimg):

    points = np.reshape(points, (40, 2), order='C')
    x = 0
    y = 1
    V = np.zeros(2)
    Nb = len(points)
    length = lgth*2
    zipped = np.zeros((Nb, 3*(length+1)))
    
    for idx in range(0,Nb):

        V[x] = points[(idx+1)%Nb,x] - points[(idx-1)%Nb,x]
        V[y] = points[(idx+1)%Nb,y] - points[(idx-1)%Nb,y]
        V = V/np.linalg.norm(V)
        Vt = V[x]
        V[x] = -V[y]
        V[y] = Vt

        P1 = np.zeros(2)
        P2 = np.zeros(2)
        P0 = np.zeros(2)
        P0[x] = points[idx,x]
        P0[y] = points[idx,y]
        P1[x] = points[idx,x] + (V[x]*length)
        P1[y] = points[idx,y] + (V[y]*length)
        P2[x] = points[idx,x] - (V[x]*length)
        P2[y] = points[idx,y] - (V[y]*length)
        
        itbuffer1 = createLineIterator(np.rint(P0).astype(int),np.rint(P1).astype(int),grayimg)
        itbuffer2 = createLineIterator(np.rint(P0).astype(int),np.rint(P2).astype(int),grayimg)        
        intensities1 = itbuffer1[:lgth+1,2]
        intensities2 = itbuffer2[:lgth+1,2]     
        xs1 = itbuffer1[:lgth+1,0]
        ys1 = itbuffer1[:lgth+1,1]
        xs2 = itbuffer2[:lgth+1,0]
        ys2 = itbuffer2[:lgth+1,1]
        xs2 = xs2[::-1]
        ys2 = ys2[::-1]
        intensities2 = intensities2[::-1]        
        xs = np.append(xs2, xs1[1:])
        ys = np.append(ys2, ys1[1:])
        intensities = np.append(intensities2, intensities1[1:])
        zipped[idx] = [val for pair in zip(xs, ys, intensities) for val in pair]
    return zipped
    
    
if __name__ == '__main__':

    radioPath = 'data/Radiographs/'
    landmarkPath = 'data/Landmarks/original/'
    sobelPath = 'data/Sobel/'

    cWhite = (255,255,255)
    cBlue = (255,0,0)
    cGreen = (0,255,0)

    k = 1 # mond
    j = 1 #tand
    data = np.loadtxt(landmarkPath+'landmarks'+str(k)+'-'+str(j)+'.txt').astype(int)
    print data,'\n'

    objectArray = np.reshape(data, (2, 40), order='F')
    print objectArray,'\n'

    points = np.reshape(data, (40, 2), order='C')
    print "points", points,'\n'

    img = cv2.imread(radioPath+'01.tif')    
    gradimg = cv2.imread(sobelPath+'01SobelGauss.png')
    #gradimg = cv2.cvtColor(gradimg, cv2.COLOR_RGB2GRAY)
    print gradimg.shape
    (h,w,_) = img.shape
    img2 = cv2.resize(img, (w/3, h/3))
    # cv2.imshow('',img2)
    # cv2.waitKey(0)

    for i in range(len(points)):
        cv2.line(img, (points[i,0],points[i,1]),(points[(i+1) % len(points),0],points[(i+1) % len(points),1]), cWhite, 2)
    img2 = cv2.resize(img, (w/3, h/3))
    cv2.imshow('',img2)
    cv2.waitKey(0)

    x = 0
    y = 1
    V = np.zeros(2)
    Nb = len(points)
    length = 20.
    for idx in range(0,Nb):

        V[x] = points[(idx+1)%Nb,x] - points[(idx-1)%Nb,x]
        V[y] = points[(idx+1)%Nb,y] - points[(idx-1)%Nb,y]
        V = V/np.linalg.norm(V)
        print "V", V
        Vt = V[x]
        V[x] = -V[y]
        V[y] = Vt

        P1 = np.zeros(2)
        P2 = np.zeros(2)
        P0 = np.zeros(2)
        P0[x] = points[idx,x]
        P0[y] = points[idx,y]
        P1[x] = points[idx,x] + (V[x]*length)
        P1[y] = points[idx,y] + (V[y]*length)
        P2[x] = points[idx,x] - (V[x]*length)
        P2[y] = points[idx,y] - (V[y]*length)
        
        grayimg = cv2.cvtColor(gradimg, cv2.COLOR_RGB2GRAY)
        itbuffer1 = createLineIterator(np.rint(P0).astype(int),np.rint(P1).astype(int),grayimg)
        itbuffer2 = createLineIterator(np.rint(P0).astype(int),np.rint(P2).astype(int),grayimg)        
        intensities1 = itbuffer1[:,2]
        intensities2 = itbuffer2[:,2]
        med1 = signal.medfilt(intensities1, 5)
        med2 = signal.medfilt(intensities2, 5)         
        # plt.plot(med)
        # plt.show()
        
        xs1 = itbuffer1[:11,0]
        ys1 = itbuffer1[:11,1]
        xs2 = itbuffer2[:11,0]
        ys2 = itbuffer2[:11,1]
        xs2 = xs2[::-1]
        ys2 = ys2[::-1]
        xs = np.append(xs2, xs1[1:])
        ys = np.append(ys2, ys1[1:])
        
        cv2.line(gradimg,tuple(np.rint(P0).astype(int)),(xs[20],ys[20]),cBlue, 2)
        cv2.line(gradimg,tuple(np.rint(P0).astype(int)),(xs[0],ys[0]),cGreen, 2)
        
        # print "P0", P0
        # print "P1", P1
        # print "P2", P2
        # print "xs1", xs1
        # print "xs2", xs2
        # print "xs", xs
        # print "ys1", ys1
        # print "ys2", ys2
        # print "ys", ys

    #img2 = cv2.resize(gradimg, (w/3, h/3))
    cv2.imshow('',gradimg[600:1400,1100:2000])
    cv2.waitKey(0)





