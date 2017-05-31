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
from auto_init import *
from radiograph import *
from debug import *
import matplotlib.colors as colors

# !DEPRECATED! This method was merely used as a test method. Run leaveOneOut() for correct execution!
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
        ratio, new_dimensions, image = scale_radiograph(image, 800)

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

# Perform segmentation, leave-one-out based approach.
def leaveOneOut():

    # Setup variabls.
    init_method = 1 # 1 for manual, 0 for auto
    test_image = 1
    name = "%02d" % test_image
    allFoundPoints1 = []
    allFoundPoints2 = []

    # some colors for drawing
    cWhite = (255,255,255)
    cBlue = (255,0,0)
    cGreen = (0,255,0)
    cGrey = (64,64,64)

    # Find all teeth.
    for tooth_to_fit in range(1,9):

        # Read image.
        image = load_image(test_image)

        # Read landmarks for tooth.
        landmark_list = load_all_landmarks_for_tooth_except_test(tooth_to_fit, test_image)
        original_lms = load_landmarks_for_person(test_image)

        # Build ASM.
        meanShape, ASM_P, eigenvalues, norm = buildASM(landmark_list)

        # Manual or auto init.
        if init_method == 1:
            init_tooth = manual_init(meanShape.get_two_lists(), norm, image)
        elif init_method == 0:
            init_tooth = auto_init(meanShape.get_two_lists(), norm, tooth_to_fit, image)

        # Scale radiograph.
        ratio, new_dimensions, image = scale_radiograph(image, 800)

        # Iterate until convergence.
        Y, foundPoints1 = iterate(tooth_to_fit, init_tooth, meanShape, ASM_P, eigenvalues, image, maxIter=5, method=1, lenProfile=5)
        Y, foundPoints2 = iterate(tooth_to_fit, init_tooth, meanShape, ASM_P, eigenvalues, image, maxIter=5, method=2, lenProfile=5, lenSearch=7)
        allFoundPoints1.append(foundPoints1)
        allFoundPoints2.append(foundPoints2)



    # # Plot found points.
    Nb = 40
    image1 = image.copy()
    image2 = image.copy()
    for idx in range(0,len(allFoundPoints1)):
        foundPointsX1, foundPointsY1 = allFoundPoints1[idx].get_two_lists(integer=True)
        foundPointsX2, foundPointsY2 = allFoundPoints2[idx].get_two_lists(integer=True)
        
        # save segmentations
        canvas1 = np.zeros(image.shape)
        canvas2 = np.zeros(image.shape)
        
        coords = np.asarray([[foundPointsX1[i],foundPointsY1[i]] for i in range(0,Nb)])
        cv2.fillConvexPoly(canvas1, coords, (64.0, 64.0, 64.0))
        cv2.imwrite(paths.FOUND+name+'-'+str(idx)+'_method1.png',canvas)
        
        coords = np.asarray([[foundPointsX2[i],foundPointsY2[i]] for i in range(0,Nb)])
        cv2.fillConvexPoly(canvas2, coords, (64.0, 64.0, 64.0))
        cv2.imwrite(paths.FOUND+name+'-'+str(idx)+'_method2.png',canvas)      

        lms, norm = original_lms[idx].scale()
        originalX, originalY = lms.get_two_lists()
        
        originalY *= ratio*norm
        originalX *= ratio*norm
        
        originalX = np.rint(originalX).astype(np.uint32)
        originalY = np.rint(originalY).astype(np.uint32) 

        for i in range(0,Nb):
            cv2.line(image1, (foundPointsX1[i],foundPointsY1[i]),(foundPointsX1[(i+1) % Nb],foundPointsY1[(i+1) % Nb]), cBlue, 2)
            cv2.line(image1, (originalX[i],originalY[i]),(originalX[(i+1) % Nb],originalY[(i+1) % Nb]), cGreen, 1)
        for i in range(0,Nb):
            cv2.line(image2, (foundPointsX2[i],foundPointsY2[i]),(foundPointsX2[(i+1) % Nb],foundPointsY2[(i+1) % Nb]), cBlue, 2)
            cv2.line(image2, (originalX[i],originalY[i]),(originalX[(i+1) % Nb],originalY[(i+1) % Nb]), cGreen, 1)
            
    # cv2.imshow('1',image1)
    cv2.imwrite(paths.FOUND+name+'_method1.png',image1)
    # cv2.imshow('2',image2)
    cv2.imwrite(paths.FOUND+name+'_method2.png',image2)
    cv2.waitKey(0)


# Util method. Only used to create images for report/presentation.    
def getAutoTeeth():
    # Read image.
    print "Reading image."
    test_image = 17
    image = load_image(test_image)
    ratio, new_dimensions, image2 = scale_radiograph(image, 800)
    # image2 = image.copy()
    teeth = []

    clrs = colors.cnames
    print clrs

    for tooth_to_fit in range(1,9):
        # Read landmarks for tooth.
        print "Reading landmarks."
        landmark_list = load_all_landmarks_for_tooth(tooth_to_fit)

        # Build ASM.
        print "Building ASM model."
        meanShape, ASM_P, eigenvalues, norm = buildASM(landmark_list)

        # Manual Init.
        print "Manual initialisation."
        # init_tooth = manual_init(meanShape.get_two_lists(), norm, image)
        init_tooth = auto_init(meanShape.get_two_lists(), norm, tooth_to_fit, image)
        teeth.append(init_tooth)
        foundPointsX1, foundPointsY1 = init_tooth.get_two_lists(integer=True)
        Nb=40
        clr = get_color(tooth_to_fit-1)
        print clr
        for i in range(0,Nb):
            cv2.line(image2, (foundPointsX1[i],foundPointsY1[i]),(foundPointsX1[(i+1) % Nb],foundPointsY1[(i+1) % Nb]), clr, 1)
    cv2.imshow('2',image2)
    # cv2.imwrite(paths.FOUND+name+'_method2.png',image2)
    cv2.waitKey(0)

# Util method. Only used to create images for report/presentation.    
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
    ev5 = eigenvalues.copy()
    ev5[2] = eigenvalues[2]*3
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev5)))
    ev6 = eigenvalues.copy()
    ev6[2] = eigenvalues[2]*10
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev6)))

    ev1 = eigenvalues.copy()
    ev1[1] = eigenvalues[1]*-10
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev1)))
    ev2 = eigenvalues.copy()
    ev2[1] = eigenvalues[1]*-3
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev2)))
    ev5 = eigenvalues.copy()
    ev5[1] = eigenvalues[1]*3
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev5)))
    ev6 = eigenvalues.copy()
    ev6[1] = eigenvalues[1]*10
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev6)))

    ev1 = eigenvalues.copy()
    ev1[0] = eigenvalues[0]*-10
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev1)))
    ev2 = eigenvalues.copy()
    ev2[0] = eigenvalues[0]*-3
    shapes.append(Landmarks(meanShape.get_list()+np.dot(ASM_P,ev2)))
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

if __name__ == '__main__':
    leaveOneOut()
