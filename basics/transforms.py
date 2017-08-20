import numpy as np

from basics import pixel_operations as po

# Just for images (uint8)
def mapValues(img, minValue, maxValue):
    width = len(img)
    height = len(img[0])
    result = np.zeros((width, height, 3), np.uint8)
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i,j])):
                result[i,j,k] = po.map_value(img[i,j,k], minValue, maxValue)
    return result

# This mapping not work if every pixel is bigger than 0
# TODO: test with scalar product when implemented.
def mapDynamicRango(img, minValue, maxValue):
    result = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i,j])):
                result[i,j,k] = po.map_dynamic_value(img[i,j,k], minValue, maxValue)
    return result

def complete_with_zeros(img, width, height):
    result = np.zeros((width, height, 3), np.uint8)
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i,j])):
                result[i,j,k] = img[i,j,k]
    return result
