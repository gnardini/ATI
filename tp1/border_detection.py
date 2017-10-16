import numpy as np
import math
from tp1 import image_operations as ops
from tp1 import umbralization as umb
from basics import transforms as tr

def _apply_mask(img, vertical_mask, horizontal_mask):
    result = np.zeros(img.shape)
    vertical = ops.apply_mask(img, vertical_mask)
    horizontal = ops.apply_mask(img, horizontal_mask)
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            for k in range(len(img[0, 0])):
                v = int(vertical[i, j, k])
                h = int(horizontal[i, j, k])
                result[i, j, k] = math.sqrt(math.pow(v, 2) + math.pow(h, 2))
    return tr.mapValues(result, np.min(result), np.max(result))

def _apply_masks(img, vertical_mask, horizontal_mask, up_left_mask, up_right_mask):
    vertical = ops.apply_mask(img, vertical_mask)
    horizontal = ops.apply_mask(img, horizontal_mask)
    up_left = ops.apply_mask(img, up_left_mask)
    up_right = ops.apply_mask(img, up_right_mask)
    result = np.zeros(img.shape)
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            for k in range(len(img[0, 0])):
                v1 = int(vertical[i, j, k])
                v2 = int(horizontal[i, j, k])
                v3 = int(up_left[i, j, k])
                v4 = int(up_right[i, j, k])
                result[i, j, k] = max([v1, v2, v3, v4])
    return tr.mapValues(result, np.min(result), np.max(result))

def prewitt(img):
    return _apply_mask(img,
                       np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
                       np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))

def sobel(img):
    return _apply_mask(img,
                       np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                       np.array([[-1,0,1], [-2,0,2], [-1,0,1]]))

def directional_prewitt(img):
    return _apply_masks(img,
                        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
                        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                        np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
                        np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]))

def directional_sobel(img):
    return _apply_masks(img,
                        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                        np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
                        np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]))

def laplace(img):
    mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    result = np.zeros(img.shape)
    mask = ops.apply_mask(img, mask)
    for i in range(1, len(img)):
        for j in range(0, len(img[0])):
            for k in range(len(img[0, 0])):
                prev = int(mask[i-1, j, k])
                v = int(mask[i, j, k])
                if (v > 0 and prev < 0) or (v < 0 and prev > 0):
                    result[i, j, k] = 255
                else:
                    result[i, j, k] = 0
    for i in range(0, len(img)):
        for j in range(1, len(img[0])):
            for k in range(len(img[0, 0])):
                prev = int(mask[i, j-1, k])
                v = int(mask[i, j, k])
                if (v > 0 and prev < 0) or (v < 0 and prev > 0):
                    result[i, j, k] = 255
    return tr.mapValues(result, np.min(result), np.max(result))

def laplace_pendiente(img, threshold):
    mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    result = np.zeros(img.shape)
    mask = ops.apply_mask(img, mask)
    for i in range(1, len(img)):
        for j in range(0, len(img[0])):
            for k in range(len(img[0, 0])):
                prev = int(mask[i - 1, j, k])
                v = int(mask[i, j, k])
                if (v > 0 and prev < 0) or (v < 0 and prev > 0):
                    result[i, j, k] = abs(v+prev)
                else:
                    result[i, j, k] = 0
    for i in range(0, len(img)):
        for j in range(1, len(img[0])):
            for k in range(len(img[0, 0])):
                prev = int(mask[i, j - 1, k])
                v = int(mask[i, j, k])
                if (v > 0 and prev < 0) or (v < 0 and prev > 0):
                    result[i, j, k] = abs(v+prev)
    img = tr.mapValues(result, np.min(result), np.max(result))
    return ops.apply_threshold(img, threshold)

def laplace_gauss(img, sigma=1, threshold=128):
    size = 4 * sigma + 1
    filter = np.zeros(size * size).reshape((size, size))
    to_border = int((size - 1) / 2)

    multiplier = -1 / (math.sqrt(2 * math.pi) * sigma * sigma * sigma)
    for x in range(-to_border, to_border + 1):
        for y in range(-to_border, to_border + 1):
            filter[to_border + x][to_border + y] =  multiplier \
                * (2 - (x*x + y*y) / (sigma*sigma)) \
                * math.exp(-(x * x + y * y) / (2 * sigma * sigma))

    mask = filter
    result = np.zeros(img.shape)
    mask = ops.apply_mask(img, mask)
    for i in range(1, len(img)):
        for j in range(0, len(img[0])):
            for k in range(len(img[0, 0])):
                prev = int(mask[i - 1, j, k])
                v = int(mask[i, j, k])
                if (v > 0 and prev < 0) or (v < 0 and prev > 0):
                    result[i, j, k] = abs(v+prev)
                else:
                    result[i, j, k] = 0
    for i in range(0, len(img)):
        for j in range(1, len(img[0])):
            for k in range(len(img[0, 0])):
                prev = int(mask[i, j - 1, k])
                v = int(mask[i, j, k])
                if (v > 0 and prev < 0) or (v < 0 and prev > 0):
                    result[i, j, k] = abs(v+prev)
    img = tr.mapValues(result, np.min(result), np.max(result))
    return ops.apply_threshold(img, threshold)

def _canny_detection(img):
    result = np.zeros(img.shape)
    angles = np.zeros(img.shape)
    vertical = ops.apply_mask(img, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    horizontal = ops.apply_mask(img, np.array([[-1,0,1], [-2,0,2], [-1,0,1]]))
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            for k in range(len(img[0, 0])):
                v = int(vertical[i, j, k])
                h = int(horizontal[i, j, k])
                result[i, j, k] = math.sqrt(math.pow(v, 2) + math.pow(h, 2))
                angles[i, j, k] = ((math.atan2(v, h) + math.pi / 2) / 2) * 360. / math.pi
                if angles[i, j, k] < 0:
                    angles[i, j, k] += math.pi
                if angles[i, j, k] < 22.5 or angles[i, j, k] > 157.5:
                    angles[i, j, k] = 0
                elif angles[i, j, k] < 67.5:
                    angles[i, j, k] = 45
                elif angles[i, j, k] < 112.5:
                    angles[i, j, k] = 90
                else:
                    angles[i, j, k] = 135
    return tr.mapValues(result, np.min(result), np.max(result)), angles

def _non_max_supression(img, angles):
    deltas = {
        0: [(1, 0), (-1, 0)],
        45: [(1, -1), (-1, 1)],
        90: [(0, 1), (0, -1)],
        135: [(-1, -1), (1, 1)],
    }
    result = np.zeros_like(img)
    for i in range(1, len(img)-1):
        for j in range(1, len(img[0])-1):
            for k in range(len(img[0, 0])):
                if img[i, j, k] != 0:
                    result[i, j, k] = img[i, j, k]
                    for delta in deltas[angles[i, j, k]]:
                        if img[i+delta[0],j+delta[1],k] > img[i,j,k]:
                            result[i,j,k] = 0
    return result


def canny_detector(img, sig1=1, sig2=3, sig3=10):
    [g1, a1] = _canny_detection(ops.apply_gauss_filter(img, sig1))
    g1 = _non_max_supression(g1, a1)
    g1 = umb.hiteresis_umbralization(g1)
    [g2, a2] = _canny_detection(ops.apply_gauss_filter(img, sig2))
    g2 = _non_max_supression(g2, a2)
    g2 = umb.hiteresis_umbralization(g2)
    # [g3, a3] = _canny_detection(ops.apply_gauss_filter(img, sig3))
    # g3 = _non_max_supression(g3, a3)
    # g3 = umb.hiteresis_umbralization(g3)
    result = np.zeros_like(g1)
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            for k in range(len(img[0, 0])):
                if g1[i,j,k] == 255 or g2[i,j,k] == 255:
                    result[i,j,k] = 255
                else:
                    result[i, j, k] = 0
    return result