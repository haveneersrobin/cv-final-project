import matplotlib.pyplot as plt
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
