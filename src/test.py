import os
import cv2

import paths
from landmarks import *

def main():
    landmar_list = load_landmarks_for_person(1)
    x = mean_shape(landmar_list)
if __name__ == '__main__':
    main()