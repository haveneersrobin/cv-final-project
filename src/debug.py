import matplotlib.pyplot as plt

def draw(shapes, color, originAtZero = False):
    """
    Debug function
    """
    for shape in shapes:
        plt.scatter(shape[0:][::2], shape[1:][::2], color=color)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.grid(True, which='both')
    if originAtZero:
        plt.axhline(y=0, color='r')
        plt.axvline(x=0, color='r')
        plt.scatter(0, 0)
    plt.show()
