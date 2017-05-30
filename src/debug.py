import matplotlib.pyplot as plt
import matplotlib.patches as patches

import collections

from landmarks import *

def draw(shapes, color = 'blue', originAtZero = False):
    """
    Debug function
    """
    if isinstance(shapes, collections.Iterable):
        for shape in shapes:
            x,y = shape.get_two_lists()
            plt.scatter(x, y, color=color)
    else:
        x,y = shapes.get_two_lists()
        plt.scatter(x, y, color=color)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.grid(True, which='both')
    if originAtZero:
        plt.axhline(y=0, color='r')
        plt.axvline(x=0, color='r')
        plt.scatter(0, 0)
    plt.show()

def draw_with_rect(shapes,rect, color = 'blue', originAtZero = False):
    fix, ax = plt.subplots(1)
    if isinstance(shapes, collections.Iterable):
        for shape in shapes:
            x,y = shape.get_two_lists()
            ax.scatter(x, y, color=color)
    else:
        x,y = shapes.get_two_lists()
        ax.scatter(x, y, color=color)
    ax.axis('equal')
    plt.gca().invert_yaxis()
    ax.grid(True, which='both')
    if originAtZero:
        ax.axhline(y=0, color='r')
        ax.axvline(x=0, color='r')
        ax.scatter(0, 0)
    rect = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3],fill=False)
    ax.add_patch(rect)
    plt.show()
