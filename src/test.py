import os
import cv2

import paths
from landmarks import *
from radiograph import *

def main():
    landmar_list = load_all_landmarks_for_tooth(1)
    images = get_all_images()
    bx, by, bw, bh = find_global_bounding_box(landmar_list)
    for im in images:
        image = cv2.imread(im)
        r, dim, resized = scale_radiograph(image, 800)
        cv2.imshow('1', resized)
        cv2.waitKey(0)

        cv2.rectangle(resized, (int(bx*r), int(by*r)), (int(r*(bx+bw)), int(r*(by+bh))), (0,255,0), 2)
        cv2.imshow('1', resized)
        cv2.waitKey(0)
        # print int(bx*r)
        # print int(bx*r+bw*r)
        # print int(by*r)
        # print int(by*r+bh*r)
        # crop = resized[int(bx*r):int(bx*r+bw*r), int(by*r):int(by*r+bh*r)]



if __name__ == '__main__':
    main()
