import paths
import cv2
import sys
from matplotlib import pyplot as plt
from ASM import *
from landmarks import *
from radiograph import *

def auto_init(model, norm, tooth, radiograph):
    # The image of the person for which we are trying to segment
    image_person = radiograph

    # Load all landmarks
    landmarks = load_all_landmarks()

    # Find position in image where to search
    maxx, minx, maxy, miny = find_extrema_list(landmarks)

    ratios = []
    templates = []

    # Create templates

    for i in range(1, 15):
        loaded = load_image(i)
        r, dim, curr_image = scale_radiograph(loaded, 800)
        rescale_lm = rescale_list(landmarks[i-1], r)
        ratios.append(r)
        teeth_template = find_global_bounding_box(rescale_lm, 5)
        cropped = cutout(curr_image, teeth_template)
        templates.append(cropped)

    # Cropped input image
    img_r, dim, curr_cropped_image = scale_radiograph(image_person, 800)
    cropped_image_person = curr_cropped_image[int(img_r*miny):int(img_r*maxy), int(img_r*minx):int(img_r*maxx)]

    # Find the best fitting template and return weighted average of offsets

    locx = 0
    locy = 0
    broke = False
    scores = [0]*14
    for index, template in enumerate(templates):
        _, w, h = template.shape[::-1]
        res = cv2.matchTemplate(cropped_image_person,template,cv2.cv.CV_TM_CCOEFF_NORMED)
        _, score, _, (x,y) = cv2.minMaxLoc(res)
        if score > 0.99:
            scores = [0]*14
            scores[index] = score
            print
            print "Near perfect match. Breaking. Score= ", score
            locx, locy = (x,y)
            broke = True
            break
        else:
            scores[index] = score
            locx += score*x
            locy += score*y
    if(not(broke)):
        locx /= 14
        locy /= 14

    # Find average offset of landmarks using the scores from above

    offset_x, offset_y = average_starting(tooth,img_r, img_r*minx+locx, img_r*miny+locy, scores)

    mx = model[0]
    my = model[1]
    mx = ((mx + abs(mx.min()))*norm*img_r)+int(img_r*minx)+locx+offset_x
    my = ((my + abs(my.min()))*norm*img_r)+int(img_r*miny)+locy+offset_y

    return Landmarks((mx, my))

# Calculate the average offset
def average_starting(tooth, img_r, minx, miny, scores):
    temp_x = 0
    temp_y = 0
    lm_tooth = load_all_landmarks_for_tooth(tooth)
    weight = sum(scores)
    for index, lm in enumerate(lm_tooth):
        lm = lm.rescale(img_r)
        x_list, y_list = lm.get_two_lists()
        temp_x += scores[index]*np.min(x_list)
        temp_y += scores[index]*np.min(y_list)
    if(minx < temp_x/weight):
        ret_x = (temp_x/weight)-minx
    else:
        ret_x = minx-(temp_x/weight)
    if(miny < temp_y/weight):
        ret_y = (temp_y/weight)-miny
    else:
        ret_y = miny-(temp_y/weight)
        
    return ret_x, ret_y

# Only for testing purposes
def main():
    lm  = load_one_landmark(3, 4)
    i = load_image(26)
    _, result = lm.to_origin()
    normalized, norm = result.scale()

    auto_init(normalized.get_two_lists(), norm,4,i)

if __name__ == '__main__':
    main()
