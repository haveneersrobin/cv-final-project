import os
import numpy as np
from debug import draw
from landmarks import *

landmarkPath = 'data/Landmarks/Original'

def findMeanShape(landmark_list):
    mean_lm = [np.zeros(landmark_list[0].get_list().shape)]
    mean_lm = mean_shape(landmark_list)
    return mean_lm

def alignShapes(lm_model, lm_target):
    """
    Returns theta and scaling factor.
    Model is the given, already scaled landmark.
    """
    modelX, modelY = lm_model.get_two_lists()
    targetX, targetY = lm_target.get_two_lists()

    cTarget = np.zeros(2)
    cTarget[0] = np.mean(targetX)
    cTarget[1] = np.mean(targetY)

    cModel = np.zeros(2)
    cModel[0] = np.mean(modelX)
    cModel[1] = np.mean(modelY)

    b = 0
    for i in xrange(0, 40):
        b += (targetX[i]*modelY[i]-targetY[i]*modelX[i])
    b /= (np.linalg.norm(lm_target.get_list())**2)

    a = np.dot(lm_model.get_list(), lm_target.get_list())/(np.linalg.norm(lm_target.get_list())**2)

    s = np.sqrt(a**2 + b**2)
    theta = np.arctan(b/a)
    t = cModel - cTarget

    return theta, s, t


def applyTransformation(x0, shape, s, theta):
    to_reshape = shape.get_list()
    objectArray = np.reshape(to_reshape, (2, 40), order='F')
    rotationMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
    rotated = np.dot(rotationMatrix, objectArray)
    scaled = s*rotated
    lm_scaled = Landmarks(scaled.T)

    x = 1.0/np.dot(x0.get_list(), lm_scaled.get_list())
    return Landmarks(x*lm_scaled.get_list())

def alignSetOfShapes(landmark_list):
    print
    print "Start aligning"
    counter = 1
    translated_lms = all_to_origin(landmark_list)
    x0,_ = translated_lms[0].scale()

    result = []

    converged = False

    while not converged:
        print "Iteration " + str(counter)
        for index, shape in enumerate(translated_lms):

            theta, s, _ = alignShapes(x0, shape)
            result.append(applyTransformation(x0, shape, s, theta))

            new_mean = findMeanShape(result)
        theta, s, _ = alignShapes(x0, new_mean)
        x0_new = applyTransformation(x0, new_mean, s, theta)
        x0_new_scaled,_ = x0_new.scale()

        if np.linalg.norm(x0_new_scaled.get_list() - x0.get_list()) < 0.02:
            converged = True
        else:
            x0 = x0_new_scaled
        counter += 1
    print "Done"
    return x0_new_scaled, result


def alignFitLandmarks(theta, s, t, newLms):
    to_reshape = newLms.get_list()
    objectArray = np.reshape(to_reshape, (2, 40), order='F')
    rotationMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
    rotated = np.dot(rotationMatrix, objectArray)
    scaled = s*rotated
    result = np.transpose(np.transpose(scaled) + np.transpose(t))
    zipped = [val for pair in zip(result[0], result[1]) for val in pair]

    return Landmarks(zipped)

def main():
    index = 0
    landmark_list = load_all_landmarks_for_tooth(1)
    #draw(landmark_list, "green")
    mean, result = alignSetOfShapes(landmark_list)
    #draw(result, "red", True)
    #alignFitLandmarks(landmark_list[1], landmark_list[2])

if __name__ == '__main__':
    main()
