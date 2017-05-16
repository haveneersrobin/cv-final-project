import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from shapely.geometry.polygon import Polygon


landmarkPath = 'data/Landmarks/Original'



'''
Given a list of x and y coordinates/points, find the center origin.
Returns the offset from the (0,0)-origina and a 2d array of points translated to origin.
NOTE: This is for one tooth
'''
def findOriginOffsetOfTooth(landmarks):
    x = landmarks[0:][::2]
    y = landmarks[1:][::2]

    xtranslated = x - np.mean(x)
    ytranslated = y - np.mean(y)

    result = [val for pair in zip(xtranslated, ytranslated) for val in pair]

    return (np.mean(x), np.mean(y)), np.asarray(result)

'''
Given a 2D-array for one kind of tooth, where each element holds a list of landmarks
it calculates the mean and returns the same list but with translated x and y coordinates.
NOTE: This is for one incisor for multiple persons
'''
def findOriginOffsetOfTeeth(landmarksList):
    cmap = mpl.cm.autumn
    result = np.zeros(landmarksList.shape)

    for index, landmark in enumerate(landmarksList):
        result[index] = findOriginOffsetOfTooth(landmark)[1]
    #     plt.scatter(result[index][0:][::2], result[index][1:][::2], color=cmap(index/float(14)))
    # plt.scatter(0,0)
    # plt.axis('equal')
    # plt.gca().invert_yaxis()
    # plt.grid(True, which='both')
    # plt.axhline(y=0, color='r')
    # plt.axvline(x=0, color='r')
    # plt.show()
    return result

def scaleLandmark(landmark):
    result = np.zeros(landmark.shape)
    norm = np.linalg.norm(landmark)
    result = landmark/norm
    return result

def main():
    # lm = np.zeros((14,80), dtype=np.float64)
    # index = 0
    # for i in range(2, 3):
    #     for file in os.listdir("./data/Landmarks/Original"):
    #             if file.endswith(str(i)+".txt"):
    #                 with open(os.path.join(landmarkPath, file), 'r') as f:
    #                     lm[index]=[line.rstrip('\n') for line in f]
    #                     index += 1
    # translatedTeeth = findOriginOffsetOfTeeth(lm)
    with open(os.path.join(landmarkPath, "landmarks1-1.txt"), 'r') as f:
        lines = [line.rstrip('\n') for line in f]
        _,translated = findOriginOffsetOfTooth(np.array(lines).astype(np.float64))
        scaled = scaleLandmark(translated)

        # plt.scatter(scaled[0:][::2], scaled[1:][::2])
        # plt.scatter(0,0)
        # plt.axis('equal')
        # plt.gca().invert_yaxis()
        # plt.grid(True, which='both')
        # plt.axhline(y=0, color='r')
        # plt.axvline(x=0, color='r')
        # plt.show()


if __name__ == '__main__':
    main()
