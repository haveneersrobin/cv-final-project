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
from landmarks import *

# Return a matrix containing the coordinates and values 
# of the pixels on the line between P1 and P2.
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
    
# Get a list of coordinates and values of pixels
# on each side of the given landmark point.    
def getNormalPoints(points, ptnNb, lgth, grayimg):

#define variables
    points = points.get_list()
    points = np.reshape(points, (40, 2), order='C')
    x = 0
    y = 1
    V = np.zeros(2)
    Nb = len(points)
    length = lgth*2
    zipped = np.zeros(3*(length+1))
    
    #define vector points
    V[x] = points[(ptnNb+1)%Nb,x] - points[(ptnNb-1)%Nb,x]
    V[y] = points[(ptnNb+1)%Nb,y] - points[(ptnNb-1)%Nb,y]
    V = V/np.linalg.norm(V)
    Vt = V[x]
    V[x] = -V[y]
    V[y] = Vt

    #define begin and end points
    P1 = np.zeros(2)
    P2 = np.zeros(2)
    P0 = np.zeros(2)
    P0[x] = points[ptnNb,x]
    P0[y] = points[ptnNb,y]
    P1[x] = points[ptnNb,x] + (V[x]*length)
    P1[y] = points[ptnNb,y] + (V[y]*length)
    P2[x] = points[ptnNb,x] - (V[x]*length)
    P2[y] = points[ptnNb,y] - (V[y]*length)
    
    pts1 = get_line(np.rint(P0[x]).astype(np.int32),np.rint(P0[y]).astype(np.int32),np.rint(P1[x]).astype(np.int32),np.rint(P1[y]).astype(np.int32))
    pts2 = get_line(np.rint(P0[x]).astype(np.int32),np.rint(P0[y]).astype(np.int32),np.rint(P2[x]).astype(np.int32),np.rint(P2[y]).astype(np.int32))
    pts = np.append(pts1[:lgth+1][::-1],pts2[1:lgth+1])
    xs = pts[0::2]
    ys = pts[1::2]  
    ints = grayimg[ys,xs]
    zipped = [val for pair in zip(xs, ys) for val in pair]
    
    return zipped, ints    
    

# Get a list of coordinates and values of pixels 
# on each side of all the landmark points.    
def getAllNormalPoints(points, lgth, grayimg):

    #define variables
    points = points.get_list()
    points = np.reshape(points, (40, 2), order='C')
    x = 0
    y = 1
    V = np.zeros(2)
    Nb = len(points)
    length = lgth*2
    zipped = np.zeros((Nb, 3*(length+1)))
    
    for idx in range(0,Nb):

        #define vector points
        V[x] = points[(idx+1)%Nb,x] - points[(idx-1)%Nb,x]
        V[y] = points[(idx+1)%Nb,y] - points[(idx-1)%Nb,y]
        V = V/np.linalg.norm(V)
        Vt = V[x]
        V[x] = -V[y]
        V[y] = Vt

        #define begin and end points
        P1 = np.zeros(2)
        P2 = np.zeros(2)
        P0 = np.zeros(2)
        P0[x] = points[idx,x]
        P0[y] = points[idx,y]
        P1[x] = points[idx,x] + (V[x]*length)
        P1[y] = points[idx,y] + (V[y]*length)
        P2[x] = points[idx,x] - (V[x]*length)
        P2[y] = points[idx,y] - (V[y]*length)

        pts1 = get_line(np.rint(P0[x]).astype(np.int32),np.rint(P0[y]).astype(np.int32),np.rint(P1[x]).astype(np.int32),np.rint(P1[y]).astype(np.int32))
        pts2 = get_line(np.rint(P0[x]).astype(np.int32),np.rint(P0[y]).astype(np.int32),np.rint(P2[x]).astype(np.int32),np.rint(P2[y]).astype(np.int32))
        pts = np.append(pts1[:lgth+1][::-1],pts2[1:lgth+1])
        xs = pts[0::2]
        ys = pts[1::2]  
        ints = grayimg[ys,xs]
        zipped[idx] = [val for pair in zip(xs, ys, ints) for val in pair]
        
    return zipped 

        
# Copied from https://stackoverflow.com/questions/25837544/get-all-points-of-a-straight-line-in-python        
def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points
    








