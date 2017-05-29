import numpy as np
import matplotlib.pyplot as plt

import os
import paths
from operator import methodcaller

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

    def get_matrix(self):
        # Returns [[x1, y2], [y1, y2], ...]
        x, y = self.get_two_lists()
        return np.asarray(zip(x,y), dtype=np.float64)

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
