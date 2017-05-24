import os
import cv2

import paths
from landmarks import *

def main():
    lm1 = Landmarks(os.path.join(paths.LANDMARK, 'landmarks1-2.txt'))
    lm2 = Landmarks(lm1.get_list())
    print lm2.get_list()
    print lm2.get_matrix()
    print lm2.get_two_lists()

if __name__ == '__main__':
    main()
