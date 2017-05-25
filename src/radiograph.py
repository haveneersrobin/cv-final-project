import np

# Takes an image and a desired height as input.
# Returns:
#   - the ratio by which width and height have to be scaled
#   - the new dimensions to use (h, w)

def scale_radiograph(image, desired_height):
    r = desired_height / image.shape[0]
    dim = (desired_height, int(image.shape[1] * r))
    
    return r, dim
