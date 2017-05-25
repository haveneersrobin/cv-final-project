import numpy as np
import os
import paths

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
                print coord_type
                print "Invalid landmarks type"

    def _open_landmarks(self, path):
        # Open landmarks and save as points if path is given
        lm_file = open(path)
        lm_vectors = np.asarray([line.rstrip('\n') for line in lm_file], dtype=np.float64)
        self.points = lm_vectors

    def get_list(self):
        # Returns [x1, y1, x2, y2 ...]
        return np.asarray(self.points, dtype=np.float64)

    def get_two_lists(self):
        # Returns two lists
        # [x1, x2, ...]
        # [y1, y2, ...]
        x = self.points[0:][::2]
        y = self.points[1:][::2]

        return np.asarray(x, dtype=np.float64),np.asarray(y, dtype=np.float64)

    def get_matrix(self):
        # Returns [[x1, y2], [y1, y2], ...]
        x, y = self.get_two_lists()
        return np.asarray(zip(x,y), dtype=np.float64)

    def _set_two_lists(self, two_lists):
        self.points = np.asarray(zip(two_lists)).flatten()

    def _set_matrix(self, matrix):
        self.points = np.asarray(matrix.flatten(), dtype=np.float64)

# Open tooth number for person. Returns landmark.
def load_one_landmark(person, tooth):
    path = os.path.join(paths.LANDMARK, 'landmarks'+str(person)+'-'+str(tooth)+'.txt')
    return Landmarks(path)

# Load all landmarks for one person. Return a list containg 8 landmark objects.
def load_landmarks_for_person(person):
    landmark_list = []
    for i in range(1, 9):
        print i
        path = os.path.join(paths.LANDMARK, 'landmarks'+str(person)+'-'+str(i)+'.txt')
        landmark_list.append(Landmarks(path))
    return landmark_list
