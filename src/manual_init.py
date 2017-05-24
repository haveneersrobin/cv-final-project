# -*- coding: utf-8 -*-

import numpy as np
import sys
import cv2
from procrustes import *

landmarkPath = 'data/Landmarks/Original'
radiographPath = 'data/Radiographs/'

def manual_init(model, radiograph):

    global current_tooth

    r = 1000.0 / radiograph.shape[0]
    dim = (1000, int(radiograph.shape[1] * r))

    scaleheight = 1000*float(dim[0])/float(radiograph.shape[0])
    scalewidth = 1000*float(dim[1])/float(radiograph.shape[1])

    resized = cv2.resize(radiograph, (dim[1], dim[0]), interpolation = cv2.INTER_AREA)

    canvas = np.array(resized)

    # Landmarks have origin (0,0) so we move the origin such that all x's and y's are positive
    # We also scale the points as we just scaled the image
    x = model[0:][::2]
    y = model[1:][::2]

    xmin = abs(x.min())
    ymin = abs(y.min())

    x_scaled = np.asarray(x+xmin, dtype=np.float64)*r*scalewidth
    y_scaled = np.asarray(y+ymin, dtype=np.float64)*r*scaleheight

    zipped = np.asarray(zip(x_scaled, y_scaled), dtype=np.int32)
    current_tooth = zipped

    global first_click
    first_click = True
    global dragged
    dragged = False
    cv2.polylines(resized, [zipped], True, (0, 255, 0))
    cv2.imshow('Place model', resized)
    cv2.setMouseCallback('Place model', mouse,canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

def mouse(event, posx, posy, flags, image):

    global currentpos
    global current_tooth
    global dragged
    global first_click

    if event == cv2.EVENT_LBUTTONDOWN:
        if not(first_click):
            dragged = True
            print dragged
            currentpos = (posx, posy)
    elif event == cv2.EVENT_LBUTTONUP:
        if not(first_click):
            dragged = False
            current_tooth = newTooth
            print current_tooth
        else:
            first_click = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if not(first_click) and dragged:
            print "dragging"
            redraw(image, posx, posy)


def redraw(image, posx, posy):
    print 'redrawing'
    global newTooth

    imgh = image.shape[0]
    tmp = np.array(image)
    dx = (posx-currentpos[0])
    dy = (posy-currentpos[1])

    points = [(p[0]+dx, p[1]+dy) for p in current_tooth]
    newTooth = points

    pimg = np.array([(int(p[0]), int(p[1])) for p in points])
    cv2.polylines(tmp, [pimg], True, (0, 255, 0))
    cv2.imshow('Place model', tmp)


def main():
    with open(os.path.join(landmarkPath, 'landmarks1-2.txt'), 'r') as f:
        i = cv2.imread(radiographPath+'01'+'.tif')
        lm = np.asarray([line.rstrip('\n') for line in f], dtype=np.float64)
        _, result = findOriginOffsetOfTooth(lm)
        normalized = scaleLandmark(result)
        manual_init(normalized, i)

if __name__ == '__main__':
    main()
