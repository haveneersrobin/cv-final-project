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
from debug import *

def main():
    print "Setup variables."
    # Setup variabls.
    person_to_fit = 1
    maxIter = 10
    allFoundPoints = []
    
    cWhite = (255,255,255)
    cBlue = (255,0,0)
    cGreen = (0,255,0)    

    for tooth_to_fit in range(1,2):
    
        # Read image.
        print "Reading image."
        image = load_image(person_to_fit)     
        
        # Read landmarks for tooth.
        print "Reading landmarks."
        landmark_list = load_all_landmarks_for_tooth(tooth_to_fit)
        original_lms = load_landmarks_for_person(person_to_fit)      
            
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
        Y, foundPoints = iterate(tooth_to_fit, init_tooth, meanShape, ASM_P, eigenvalues, image, maxIter)
        allFoundPoints.append(foundPoints)
        
             
        
    # # Plot found points.
    Nb = 40
    for idx in range(0,len(allFoundPoints)):
        foundPointsX, foundPointsY = allFoundPoints[idx].get_two_lists(integer=True)   
        lms, norm = original_lms[idx].scale()
        originalX, originalY = lms.get_two_lists()
        originalY *= ratio*norm
        originalX *= ratio*norm
        originalX = np.rint(originalX).astype(np.uint32)
        originalY = np.rint(originalY).astype(np.uint32)
        for i in range(0,Nb):
            cv2.line(image, (foundPointsX[i],foundPointsY[i]),(foundPointsX[(i+1) % Nb],foundPointsY[(i+1) % Nb]), cBlue, 1)
        for i in range(0,Nb):
            cv2.line(image, (originalX[i],originalY[i]),(originalX[(i+1) % Nb],originalY[(i+1) % Nb]), cGreen, 1)            
    cv2.imshow('',image)
    cv2.waitKey(0)       


def leaveOneOut():

    print "Setup variables."
    # Setup variabls.
    maxIter = 10
    test_image = 1
    allFoundPoints = []
    
    cWhite = (255,255,255)
    cBlue = (255,0,0)
    cGreen = (0,255,0)    

    for tooth_to_fit in range(1,2):
    
        # Read image.
        print "Reading image."
        image = load_image(test_image)     
        
        # Read landmarks for tooth.
        print "Reading landmarks."
        landmark_list = load_all_landmarks_for_tooth_except_test(tooth_to_fit, test_image)
        original_lms = load_landmarks_for_person(test_image)      
            
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
        Y, foundPoints = iterate(tooth_to_fit, init_tooth, meanShape, ASM_P, eigenvalues, image, maxIter)
        allFoundPoints.append(foundPoints)
        
           
        
    # # Plot found points.
    Nb = 40
    for idx in range(0,len(allFoundPoints)):
        foundPointsX, foundPointsY = allFoundPoints[idx].get_two_lists(integer=True)   
        lms, norm = original_lms[idx].scale()
        originalX, originalY = lms.get_two_lists()
        originalY *= ratio*norm
        originalX *= ratio*norm
        originalX = np.rint(originalX).astype(np.uint32)
        originalY = np.rint(originalY).astype(np.uint32)
        for i in range(0,Nb):
            cv2.line(image, (foundPointsX[i],foundPointsY[i]),(foundPointsX[(i+1) % Nb],foundPointsY[(i+1) % Nb]), cBlue, 1)
        for i in range(0,Nb):
            cv2.line(image, (originalX[i],originalY[i]),(originalX[(i+1) % Nb],originalY[(i+1) % Nb]), cGreen, 1)            
    cv2.imshow('',image)
    cv2.waitKey(0)    

    return allFoundPoints
    
def meanshapevariance():

    # Read landmarks for tooth.
    print "Reading landmarks."
    landmark_list = load_all_landmarks_for_tooth(1)
        
    # Build ASM.
    print "Building ASM model."
    meanShape, ASM_P, eigenvalues, norm = buildASM(landmark_list)
    
    draw(Landmarks(meanShape.get_list()+np.dot(ASM_P,eigenvalues)))
    
    
    shapes = [Landmarks(meanShape.get_list()+np.dot(ASM_P,eigenvalues))]
    
    ev1 = eigenvalues.copy()
    ev1[2] = eigenvalues[2]*-10
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev1)))
    ev2 = eigenvalues.copy()
    ev2[2] = eigenvalues[2]*-3
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev2)))
    # ev3 = eigenvalues.copy()
    # ev3[1] = eigenvalues[1]*-2
    # shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev3)))    
    # ev4 = eigenvalues.copy()
    # ev4[1] = eigenvalues[1]*2
    # shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev4)))
    ev5 = eigenvalues.copy()
    ev5[2] = eigenvalues[2]*3
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev5)))    
    ev6 = eigenvalues.copy()
    ev6[2] = eigenvalues[2]*10
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev6))) 
    
    # draw(shapes)        
    
    ev1 = eigenvalues.copy()
    ev1[1] = eigenvalues[1]*-10
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev1)))
    ev2 = eigenvalues.copy()
    ev2[1] = eigenvalues[1]*-3
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev2)))
    # ev3 = eigenvalues.copy()
    # ev3[1] = eigenvalues[1]*-2
    # shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev3)))    
    # ev4 = eigenvalues.copy()
    # ev4[1] = eigenvalues[1]*2
    # shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev4)))
    ev5 = eigenvalues.copy()
    ev5[1] = eigenvalues[1]*3
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev5)))    
    ev6 = eigenvalues.copy()
    ev6[1] = eigenvalues[1]*10
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev6))) 
    
    # draw(shapes)      
    
    ev1 = eigenvalues.copy()
    ev1[0] = eigenvalues[0]*-10
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev1)))
    ev2 = eigenvalues.copy()
    ev2[0] = eigenvalues[0]*-3
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev2)))
    # ev3 = eigenvalues.copy()
    # ev3[0] = eigenvalues[0]*-2
    # shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev3)))    
    # ev4 = eigenvalues.copy()
    # ev4[0] = eigenvalues[0]*2
    # shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev4)))
    ev5 = eigenvalues.copy()
    ev5[0] = eigenvalues[0]*3
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev5)))    
    ev6 = eigenvalues.copy()
    ev6[0] = eigenvalues[0]*10
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev6))) 
    
    
    mx,my = meanShape.get_two_lists()
    mx = np.append(mx, mx[0])
    my = np.append(my, my[0])
    
    # draw(shapes)    
    fig = plt.figure()
    
    s1 = fig.add_subplot(131)
    s1.set_title('Mode 1')
    
    for i in range(9,13):
        x,y = shapes[i].get_two_lists()
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        s1.plot(x, y, linewidth=.5, color='r') 
        
    s1.plot(mx, my, linewidth=1, color='b')    
    plt.gca().invert_yaxis()
    plt.axis('equal')
    # plt.axis('off')

    s2 = fig.add_subplot(132)
    s2.set_title('Mode 2')
    
    for i in range(5,9):
        x,y = shapes[i].get_two_lists()
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        s2.plot(x, y, linewidth=.5, color='r') 

    s2.plot(mx, my, linewidth=1, color='b')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    # plt.axis('off')

    s3 = fig.add_subplot(133)
    s3.set_title('Mode 3')
    
    for i in range(1,5):
        x,y = shapes[i].get_two_lists()
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        s3.plot(x, y, linewidth=.5, color='r')   
    
    s3.plot(mx, my, linewidth=1, color='b')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    # plt.axis('off')
    
    # plt.title('Primary modes variation')
    plt.show()
    
def validateResults(allFoundPoints, originalLms):

    for idx, lm in enumerate(allFoundPoints):
        originals = originalLms[idx]
        
    
    
    
if __name__ == '__main__':
    main()
