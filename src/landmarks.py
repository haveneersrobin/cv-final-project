import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
import paths
from operator import methodcaller
from debug import *

import collections

class Landmarks:

    def __init__(self, coordinates):

        ## Landmarks can be initialized with
        ## - the path to the landmarks
        ## - the coordinates as one list [x1, y1, x2, y2 ...]
        ## - the coordinates as matrix [[x1, y1], [x2, y2] ...]
        self.points = np.zeros(80)
        if coordinates is not(None):
            if isinstance(coordinates, str):
                self._open_landmarks(coordinates)
            elif isinstance(coordinates, tuple):
                self._set_two_lists(coordinates)
            elif isinstance(coordinates, np.ndarray):
                if coordinates.shape == (80,):
                    self.points = coordinates
                elif coordinates.shape == (40,2):
                    self._set_matrix(coordinates)
            else:
                print "Invalid landmarks type"
                print type(coordinates)
                print coordinates

    def _open_landmarks(self, path):
        # Open landmarks and save as points if path is given
        lm_file = open(path)
        lm_vectors = np.asarray([line.rstrip('\n') for line in lm_file], dtype=np.float64)
        self.points = lm_vectors

    def get_list(self):
        # Returns [x1, y1, x2, y2 ...]
        return np.asarray(self.points, dtype=np.float64)

    def get_two_lists(self,integer=False):
        # Returns two lists
        # [x1, x2, ...]
        # [y1, y2, ...]
        x = self.points[0:][::2]
        y = self.points[1:][::2]
        if not integer:
            return np.asarray(x, dtype=np.float64),np.asarray(y, dtype=np.float64)
        else:
            return np.asarray(x, dtype=np.int32),np.asarray(y, dtype=np.int32)

    def get_matrix(self, cv=False):
        # Returns [[x1, y2], [y1, y2], ...]
        x, y = self.get_two_lists()
        if not cv:
            return np.asarray(zip(x,y), dtype=np.float64)
        else:
            result = [[[xco, yco]] for xco in x for yco in y]
            return np.asarray(result, np.float32)

    def _set_two_lists(self, two_lists):
        self.points = np.asarray(zip(two_lists[0], two_lists[1])).flatten()

    def _set_matrix(self, matrix):
        self.points = np.asarray(matrix.flatten(), dtype=np.float64)

    def scale(self):
        self_list = self.get_list()
        norm = self.get_norm()
        result = self_list/norm
        return Landmarks(result), norm

    def to_origin(self):
        x,y = self.get_two_lists()

        xtranslated = x - np.mean(x)
        ytranslated = y - np.mean(y)

        return (np.mean(x), np.mean(y)), Landmarks((xtranslated,ytranslated))

    def get_norm(self):
        self_list = self.get_list()
        return np.linalg.norm(self_list)

    # Dot product for landmarks
    def dot(self, other):
        return Landmarks(np.dot(self.get_list(), other.get_list()))

    # Rescale landmarks with given ratio
    def rescale(self, ratio):
        x,y = self.get_two_lists()
        x_scale = x * ratio
        y_scale = y * ratio
        return Landmarks((x_scale, y_scale))

    def find_extrema(self):
        x,y = self.get_two_lists()
        return max(x), min(x), max(y), min(y)





# Open one tooth for one person. Returns landmark.
def load_one_landmark(person, tooth):
    print "Opening tooth " + str(tooth) + " for person " +str(person) + "."
    path = os.path.join(paths.LANDMARK, 'landmarks'+str(person)+'-'+str(tooth)+'.txt')
    return Landmarks(path)

# Open landmarks for given tooth for all persons. Returns list of landmarks of one tooth for all persons.
def load_all_landmarks_for_tooth(tooth):
    print "Opening all landmarks of tooth " + str(tooth) + "."
    print "Opening person",
    landmark_list = []
    for i in range(1, 15):
        print str(i) + "...",
        path = os.path.join(paths.LANDMARK, 'landmarks'+str(i)+'-'+str(tooth)+'.txt')
        landmark_list.append(Landmarks(path))
    return landmark_list

# Open landmarks for given tooth for all persons. Returns list of landmarks of one tooth for all persons.
def load_all_landmarks_for_tooth_except_test(tooth, test):
    print "Opening all landmarks of tooth " + str(tooth) + "."
    print "Opening person",
    landmark_list = []
    for i in range(1, 15):
        if not test == i:
            print str(i) + "...",
            path = os.path.join(paths.LANDMARK, 'landmarks'+str(i)+'-'+str(tooth)+'.txt')
            landmark_list.append(Landmarks(path))
    return landmark_list    
    
# Load all landmarks for one person. Return a list containg 8 landmark objects.
def load_landmarks_for_person(person):
    print "Opening all landmarks for person " + str(person) + "."
    print "Opening tooth",
    landmark_list = []
    for i in range(1, 9):
        print str(i) + "...",
        path = os.path.join(paths.LANDMARK, 'landmarks'+str(person)+'-'+str(i)+'.txt')
        landmark_list.append(Landmarks(path))
    return landmark_list

def load_all_landmarks():
    print "Opening all landmarks "
    landmarks_list = []
    for i in range(1, 15):
        landmarks = load_landmarks_for_person(i)
        landmarks_list.append(landmarks)
    return landmarks_list

# Given a list of landmark objects
# it calculates the mean and returns a list with new landmarks
# but with translated x and y coordinates.
def all_to_origin(landmarks_list):
    result = []
    for index, landmark in enumerate(landmarks_list):
        _, out = landmark.to_origin()
        result.append(out)
    return result

# Return the mean landmark given a list of landmarks
def mean_shape(landmarks_list):
    return Landmarks(np.mean(map(methodcaller('get_list'), landmarks_list), axis=0))

# Given a list of landmarks, finds a bounding box that is the smallest that contains all the landmarks
def find_global_bounding_box(landmarks_list, error=0):
    points = []
    for lm in landmarks_list:
        rect = cv2.boundingRect(lm.get_matrix(cv=True))
        points.append([[rect[0]-error, rect[1]-error]])
        third = rect[0]+rect[2]+error
        fourth = rect[1]+rect[3]+error
        points.append([[third, fourth]])
    return cv2.boundingRect(np.asarray(points, np.int32))


def find_extrema(landmark_list):
    maxx = 0.0
    maxy = 0.0
    minx = float("inf")
    miny = float("inf")

    for lm in landmark_list:
        tmaxx, tminx, tmaxy, tminy = lm.find_extrema()
        if tmaxx > maxx:
            maxx = tmaxx
        if tmaxy > maxy:
            maxy = tmaxy
        if tminx < minx:
            minx = tminx
        if tminy < miny:
            miny = tminy

    return int(maxx), int(minx), int(maxy), int(miny)

def find_extrema_list(list_of_landmark_lists):
    maxx = 0.0
    maxy = 0.0
    minx = float("inf")
    miny = float("inf")
    for lm_list in list_of_landmark_lists:
        tmaxx, tminx, tmaxy, tminy = find_extrema(lm_list)
        if tmaxx > maxx:
            maxx = tmaxx
        if tmaxy > maxy:
            maxy = tmaxy
        if tminx < minx:
            minx = tminx
        if tminy < miny:
            miny = tminy
    return int(maxx), int(minx), int(maxy), int(miny)

def rescale_list(landmark_list, ratio):
    result = []
    for lm in landmark_list:
        result.append(lm.rescale(ratio))
    return result

def calc_average_of_lm_list(landmark_list):
    print landmark_list
