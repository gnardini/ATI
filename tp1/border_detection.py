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

def _calculate_angle(v, h):
    if h == 0:
        angle = 90
    else:
        angle = math.atan2(v, h) * 180. / math.pi
    if angle < 0:
        angle += 180
    if angle < 22.5 or angle > 157.5:
        return 0
    elif angle < 67.5:
        return 45
    elif angle < 112.5:
        return 90
    else:
        return 135

def _canny_detection(img):
    deltas = {
        0: [(1, 0), (-1, 0)],
        45: [(-1, -1), (1, 1)],
        90: [(0, 1), (0, -1)],
        135: [(1, -1), (-1, 1)],
    }
    result = np.zeros(img.shape)
    vertical = ops.apply_mask(img, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    horizontal = ops.apply_mask(img, np.array([[-1,0,1], [-2,0,2], [-1,0,1]]))
    s = sobel(img)
    for i in range(1, len(img)-1):
        for j in range(1, len(img[0])-1):
            for k in range(len(img[0, 0])):
                h = int(vertical[i, j, k])
                v = int(horizontal[i, j, k])
                angle = _calculate_angle(v, h)
                result[i, j, k] = s[i, j, k]
                for delta in deltas[angle]:
                    if s[i+delta[0],j+delta[1],k] > s[i,j,k]:
                        result[i,j,k] = 0
    return tr.mapValues(result, np.min(result), np.max(result))

def canny_detector(img, sig1=1, sig2=3):
    g1 = _canny_detection(ops.apply_gauss_filter(img, sig1))
    g1 = umb.hiteresis_umbralization(g1)
    return g1

    # g2 = _canny_detection(ops.apply_gauss_filter(img, sig2))
    # g2 = umb.hiteresis_umbralization(g2)

    # result = np.zeros_like(g1)
    # for i in range(0, len(img)):
    #     for j in range(0, len(img[0])):
    #         for k in range(len(img[0, 0])):
    #             if g1[i,j,k] == 255 and g2[i,j,k] == 255:
    #                 result[i,j,k] = 255
    #             else:
    #                 result[i, j, k] = 0
    # return result

def susan(img):
    mask = [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ]
    t = 27
    rounding_tolerance = .1
    to_border = int((len(mask) - 1) / 2)
    result = np.copy(img)
    for i in range(to_border, len(img) - to_border):
        for j in range(to_border, len(img[0]) - to_border):
            for k in range(len(img[0, 0])):
                similar = 0
                for x in range(-to_border, to_border + 1):
                    for y in range(-to_border, to_border + 1):
                        if mask[x + to_border][y + to_border] == 1:
                            if abs(int(img[i,j,k])-int(img[i+x, j+y, k])) >= t:
                                similar += 1
                s = 1 - (similar / 37)
                if .5-rounding_tolerance < s < .5+rounding_tolerance:
                    result[i, j] = [0, 255, 0]
                if .25-rounding_tolerance < s < .25+rounding_tolerance:
                    result[i, j] = [255, 0, 0]
    return result

def _find_white_points(img):
    points = []
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if img[i, j, 0] == 255:
                points.append((i, j))
    return points

def _satisfies_line_normal_equation(x, y, tita, ro, epsilon):
    return abs(ro - x * math.cos(tita) - y * math.sin(tita)) < epsilon

def _draw_line(img, tita, ro, epsilon):
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if _satisfies_line_normal_equation(i, j, tita, ro, epsilon):
                img[i,j] = [128, 128, 128]


def hough_transform(img, ro_steps=500, tita_steps=360, epsilon=1):
    result = np.zeros_like(img)
    acum = np.zeros((tita_steps, ro_steps))

    tita2 = math.pi / 2
    tita1 = -tita2
    tita_step = (tita2 - tita1) / (tita_steps - 1)

    ro2 = max(len(img), len(img[0])) * math.sqrt(2)
    ro1 = -ro2
    ro_step = (ro2 - ro1) / (ro_steps - 1)

    current_tita = tita1
    current_tita_step = 0

    newImg = umb.global_thresholding(prewitt(img))
    white_points = _find_white_points(newImg)

    while current_tita_step < tita_steps:
        current_ro = ro1
        current_ro_step = 0

        while current_ro_step < ro_steps:
            for point in white_points:
                if _satisfies_line_normal_equation(point[0], point[1], current_tita, current_ro, epsilon):
                    acum[current_tita_step][current_ro_step] += 1

            current_ro_step += 1
            current_ro += ro_step

        current_tita_step += 1
        current_tita += tita_step

    _max = np.max(acum)
    thresh = .75 * _max

    current_tita = tita1
    current_tita_step = 0

    while current_tita_step < tita_steps:
        current_ro = ro1
        current_ro_step = 0

        while current_ro_step < ro_steps:
            if (acum[current_tita_step][current_ro_step] > thresh):
                _draw_line(result, current_tita, current_ro, epsilon)

            current_ro_step += 1
            current_ro += ro_step

        current_tita_step += 1
        current_tita += tita_step

    return result
