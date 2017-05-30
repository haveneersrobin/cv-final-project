# -*- coding: utf-8 -*-
import numpy as np
import sys
import cv2
import paths

from landmarks import *
from procrustes import *
from debug import *
from radiograph import *

def manual_init(model, norm, radiograph):

    global current_tooth
    global first_click
    global dragged

    first_click = True
    dragged = False
    ratio, new_dimensions, resized = scale_radiograph(radiograph, 800)

    canvas = np.array(resized)

    # Landmarks have origin (0,0) so we move the origin such that all x's and y's are positive
    # We also scale the points as we just scaled the image
    x = model[0]
    y = model[1]

    xmin = abs(x.min())
    ymin = abs(y.min())

    x_scaled = np.asarray(x+xmin, dtype=np.float64)*ratio*norm
    y_scaled = np.asarray(y+ymin, dtype=np.float64)*ratio*norm

    zipped = np.asarray(zip(x_scaled, y_scaled), dtype=np.int32)
    print zipped
    current_tooth = zipped

    cv2.polylines(resized, [zipped], True, (0, 255, 0))
    cv2.imshow('Place model', resized)
    cv2.setMouseCallback('Place model', mouse,canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print current_tooth
    return Landmarks(current_tooth)


def mouse(event, posx, posy, flags, image):

    global currentpos
    global current_tooth
    global newTooth
    global dragged
    global first_click

    if event == cv2.EVENT_LBUTTONDOWN:
        if not(first_click):
            dragged = True
            currentpos = (posx, posy)
    elif event == cv2.EVENT_LBUTTONUP:
        if not(first_click):
            dragged = False
            current_tooth = newTooth
        else:
            first_click = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if not(first_click) and dragged:
            redraw(image, posx, posy)


def redraw(image, posx, posy):
    global newTooth

    imgh = image.shape[0]
    tmp = np.array(image)
    dx = (posx-currentpos[0])
    dy = (posy-currentpos[1])

    points = np.asarray([(point[0]+dx, point[1]+dy) for point in current_tooth], dtype=np.int32)
    newTooth = points

    cv2.polylines(tmp, [points], True, (0, 255, 0))
    cv2.imshow('Place model', tmp)


def main():
        lm  = load_one_landmark(1, 2)
        i = load_image(1)
        _, result = lm.to_origin()
        normalized, norm = result.scale()

        manual_init(normalized.get_two_lists(), norm, i)

if __name__ == '__main__':
    main()
