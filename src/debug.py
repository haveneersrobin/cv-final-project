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
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            plt.plot(x, y, linewidth=.5, color=color)
    else:
        x,y = shapes.get_two_lists()
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        plt.plot(x, y, linewidth=.5, color=color)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    # plt.axis('off')
    # plt.grid(True, which='both')
    if originAtZero:
        plt.axhline(y=0, color='b', linewidth = .5)
        plt.axvline(x=0, color='b', linewidth = .5)
        # plt.scatter(0, 0)
    plt.show()
