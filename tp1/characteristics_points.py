import numpy as np
import math
from tp1 import image_operations as ops
from tp1 import umbralization as umb
from basics import transforms as tr
from tp1 import border_detection as bd

def harris(img):
    porcentage = 0.05
    k = 0.04
    ## Apply prewitt masks
    vertical_mask = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    horizontal_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical = ops.apply_mask(img, vertical_mask)
    horizontal = ops.apply_mask(img, horizontal_mask)
    ## Calculate components from masks
    vertical_2 = vertical * vertical

    horizontal_2 = horizontal * horizontal
    vertical_horizontal = vertical * vertical
    ## Apply Gauss filter
    vertical_2 = ops.apply_gauss_filter(vertical_2, 2)
    horizontal_2 = ops.apply_gauss_filter(horizontal_2, 2)
    vertical_horizontal = ops.apply_gauss_filter(vertical_horizontal, 2)

    part1 = vertical_horizontal * vertical_horizontal
    part2 = vertical_2 + horizontal_2

    cim =(horizontal_2 * vertical_2 - part1 - (part2 * part2) * k)
    maxCim = cim.max();

    #Add result to original image
    result = np.copy(img)
    for i in range(len(result)):
        for j in range(len(result[i])):
            if (cim[i,j,0] > porcentage*maxCim) or (cim[i,j,1]> porcentage*maxCim) or (cim[i,j,2]> porcentage*maxCim):
                result[i,j,0] = 255
                result[i,j,1] = 0
                result[i,j,2] = 0
    return result
