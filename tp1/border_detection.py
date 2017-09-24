import numpy as np
import math
from tp1 import image_operations as ops
from basics import transforms as tr

def sobel(img):
    result = np.zeros(img.shape)
    vertical = ops.apply_mask(img, np.array([[-1,-1,-1], [0,0,0], [1,1,1]]))
    horizontal = ops.apply_mask(img, np.array([[-1,0,1], [-1,0,1], [-1,0,1]]))
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            for k in range(len(img[0, 0])):
                v = int(vertical[i, j, k])
                h = int(horizontal[i, j, k])
                result[i, j, k] = math.sqrt(v*v+h*h)
    return tr.mapValues(result, np.min(result), np.max(result))
