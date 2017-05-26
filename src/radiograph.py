import cv2
import paths

# Takes an image and a desired height as input.
# Returns:
#   - the ratio by which width and height have to be scaled
#   - the new dimensions to use (h, w)
def scale_radiograph(image, desired_height):
    r = float(desired_height) / image.shape[0]
    dim = (desired_height, int(image.shape[1] * r))

    return r, dim
    
# Convert an RGB image to grayscale.
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Returns the radiograph for the given person
def load_image(person):
    string = "%02d" % (person)
    return cv2.imread(paths.RADIO+string+'.tif')
