import os
import numpy as np
from debug import draw

landmarkPath = 'data/Landmarks/Original'


def findOriginOffsetOfTooth(landmarks):
    """
    Given a list of x and y coordinates/points, find the center origin.
    Returns the offset from the (0,0)-origina and a 2d array of
    points translated to origin.
    NOTE: This is for one tooth
    """
    x = landmarks[0:][::2]
    y = landmarks[1:][::2]

    xtranslated = x - np.mean(x)
    ytranslated = y - np.mean(y)

    result = [val for pair in zip(xtranslated, ytranslated) for val in pair]

    return (np.mean(x), np.mean(y)), np.asarray(result)

def findMeanShape(shapeList):
    mean_shape = np.zeros(shapeList[0].shape)
    mean_shape = np.mean(shapeList, axis=0)
    return mean_shape

def findOriginOffsetOfTeeth(landmarksList):
    """
    Given a 2D-array for one kind of tooth,
    where each element holds a list of landmarks
    it calculates the mean and returns the same list
    but with translated x and y coordinates.
    NOTE: This is for one incisor for multiple persons
    """
    result = np.zeros(landmarksList.shape)
    for index, landmark in enumerate(landmarksList):
        result[index] = findOriginOffsetOfTooth(landmark)[1]

    return result


def scaleLandmark(landmark):
    result = np.zeros(landmark.shape)
    norm = np.linalg.norm(landmark)
    result = landmark/norm
    return result


def alignShapes(model, target):
    """
    Returns theta and scaling factor.
    Model is the given, already scaled landmark.
    """
    modelX = model[0:][::2]
    modelY = model[1:][::2]

    targetX = target[0:][::2]
    targetY = target[1:][::2]
    
    cTarget = np.zeros(2)
    cTarget[0] = np.mean(targetX)
    cTarget[1] = np.mean(targetY)
    
    cModel = np.zeros(2)
    cModel[0] = np.mean(modelX)
    cModel[1] = np.mean(modelY)

    b = 0
    for i in xrange(0, 40):
        b += (targetX[i]*modelY[i]-targetY[i]*modelX[i])
    b /= (np.linalg.norm(target)**2)

    a = np.dot(model, target)/(np.linalg.norm(target)**2)

    s = np.sqrt(a**2 + b**2)
    theta = np.arctan(b/a)
    t = cModel - cTarget

    return theta, s, t


def applyTransformation(x0, shape, s, theta):
    objectArray = np.reshape(shape, (2, 40), order='F')
    rotationMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
    rotated = np.dot(rotationMatrix, objectArray)
    scaled = s*rotated
    
    zipped = [val for pair in zip(scaled[0], scaled[1]) for val in pair]
    x = 1.0/np.dot(x0, zipped)
    return x*np.asarray(zipped)

def alignSetOfShapes(setOfShapes):
    translatedShapes = findOriginOffsetOfTeeth(setOfShapes)
    x0 = scaleLandmark(translatedShapes[0])

    result = np.zeros(setOfShapes.shape)

    converged = False

    while not converged:
        print "running"
        for index, shape in enumerate(translatedShapes):
            theta, s, _ = alignShapes(x0, shape)
            result[index] = applyTransformation(x0, shape, s, theta)

        new_mean = findMeanShape(result)
        theta, s, _ = alignShapes(x0, new_mean)
        x0_new = applyTransformation(x0, new_mean, s, theta)
        x0_new_scaled = scaleLandmark(x0_new)
        if np.linalg.norm(x0_new_scaled - x0) < 0.02:
            converged = True
        else:
            x0 = x0_new_scaled
    return x0_new_scaled, result
    
def alignFitLandmarks(theta, s, t, newLms):    
    theta = -theta
    objectArray = np.reshape(newLms, (2, 40), order='F')
    rotationMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
    rotated = np.dot(rotationMatrix, objectArray)
    scaled = (1./s)*rotated
    result = np.transpose(np.transpose(scaled) + np.transpose(t))
    zipped = [val for pair in zip(result[0], result[1]) for val in pair]

    return zipped

def main():
    lm = np.zeros((14, 80), dtype=np.float64)
    index = 0
    for file in os.listdir("./data/Landmarks/Original"):
        if file.endswith("1.txt"):
            with open(os.path.join(landmarkPath, file), 'r') as f:
                lm[index] = [line.rstrip('\n') for line in f]
                index += 1
    draw(lm, "green")
    mean, result = alignSetOfShapes(lm)
    draw(result, "red", True)
    alignFitLandmarks(lm[1], lm[2])

if __name__ == '__main__':
    main()
