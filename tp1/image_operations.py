import cv2
import numpy as np
import math

import matplotlib as plt
plt.use("TkAgg")
import matplotlib.pyplot

from basics import transforms as tr
from basics import pixel_operations as po

def add_images(img1, img2):
    return _apply_between_images(po.sum, img1, img2)

def subtract_images(img1, img2):
    return _apply_between_images(po.subtract, img1, img2)

def multiply_images(img1, img2):
    return _apply_between_images(po.multiply, img1, img2)

def multiply_by_scalar(img, scalar=1.5):
    width = len(img)
    height = len(img[0])
    result = np.zeros((width, height, 3), np.int32)
    extremeValue = 0
    for i in range(width):
        for j in range(height):
            for k in range(len(img[i,j])):
                result[i,j,k] = np.int32(img[i,j,k]) * scalar
                extremeValue = max(result[i,j,k], extremeValue)
    result = tr.mapDynamicRango(result, extremeValue)
    return result

#TODO: extract min and max to separate func and with different bands
def compress_dynamic_range(img):
    extremeValue = 0
    width = len(img)
    height = len(img[0])
    for i in range(width):
        for j in range(height):
            for k in range(len(img[i,j])):
                extremeValue = max(result[i,j,k], extremeValue)
    return tr.mapDynamicRango(img, extremeValue)

def apply_gamma_potential(img, gamma=2):
    result = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i,j])):
                result[i,j,k] = round(math.pow(255,(1-gamma))*math.pow(img[i,j,k], gamma))
    return result

def negative(img):
    result = np.copy(img)
    return np.vectorize(lambda x: 255 - x)(result).astype(np.uint8)

def grayscale_histogram(img, show_plot=True):
    colorsCount = np.zeros(256)
    width = len(img)
    height = len(img[0])
    for i in range(width):
        for j in range(height):
            colorsCount[img[i,j,0]] += 1
    for x in range(len(colorsCount)):
        colorsCount[x] /= (width * height)
        matplotlib.pyplot.bar(x, colorsCount[x], 1, color="#3292e1")
    if show_plot:
        plt.pyplot.show()
    return colorsCount

def increase_contrast(img):
    result = np.copy(img)
    for k in range(len(img[0][0])):
        matrix = img[:,:,k]
        mean = np.mean(matrix)
        std = np.std(matrix)
        std = np.sqrt(std)
        if mean - std < 0:
            r1 = mean // 2
        else:
            r1 = mean - std
        if mean + std > 255:
            r2 = mean + (255 - mean) // 2
        else:
            r2 = mean + std
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i,j] < r1:
                    result[i,j,k] = matrix[i][j] // 2
                elif matrix[i,j] > r2:
                    result[i, j, k] = matrix[i,j] + (255 - matrix[i,j]) // 2
                else:
                    result[i,j,k] = matrix[i,j]

    return result

def apply_threshold(img, threshold):
    result = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i,j])):
                result[i,j,k] = 0 if img[i,j,k] < threshold else 255
    return result

# TODO: arregalr estilo de esto.
def equalize(img):
    hist = _hist(img, len(img), len(img[0]))
    cdf = _cdf(hist, len(img) * len(img[0]))
    smin = _cdf_min(cdf)
    result = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i,j])):
                result[i,j,k] = round( (cdf[img[i,j,k], k] - smin[k]) / (1 - smin[k]) * 255 )
    return result

def add_gaussian_noise(img, percent=.2, mean=0, stdv=10):
    total = len(img) * len(img[0])
    noise = np.random.normal(mean, stdv, int(total * percent))
    return _add_noise(img, noise, 'add')

def add_rayleigh_noise(img, percent=.2, scale=.5):
    total = len(img) * len(img[0])
    noise = np.random.rayleigh(scale, int(total * percent))
    noise = np.sort(noise)
    return _add_noise(img, noise, 'mult')

def add_exponential_noise(img, percent=.2, _lambda=1):
    total = len(img) * len(img[0])
    noise = np.random.exponential(1/_lambda, int(total * percent))
    return _add_noise(img, noise, 'mult')

def add_salt_pepper_noise(img, percent=.2):
    p0 = percent/2
    p1 = 1-p0
    result = np.copy(img)
    transformed = np.vectorize(lambda x: transform_salt_pepper(x, p0, p1))(result[:,:,0]).astype(np.uint8)
    for k in range(len(img[0,0])):
        result[:,:,k] = transformed
    return result

def transform_salt_pepper(px, p0, p1):
    r = np.random.random()
    if r < p0:
        return 0
    elif r > p1:
        return 255
    else:
        return px

def _add_noise(img, noise, mode='add'):
    total = len(img) * len(img[0])
    zeros = np.zeros(total - len(noise))
    if mode == 'mult':
        zeros = np.ones(total - len(noise))
    noise = np.concatenate((noise, zeros))
    np.random.shuffle(noise)
    noise = noise.reshape((len(img), len(img[0])))
    result = np.copy(img).astype(np.float32)
    for k in range(len(img[0, 0])):
        if mode == 'add':
            result[:, :, k] += noise
        elif mode == 'mult':
            result[:, :, k] *= noise
    return tr.mapValues(result, np.min(result), np.max(result))

def apply_mean_filter(img, size=3):
    if size%2 == 0:
        print('Tiene que ser impar')
        return img
    filter = np.ones(size*size).reshape((size, size)) / (size*size)
    to_border = int((size-1)/2)
    result = img.copy()
    for i in range(to_border, len(img)-to_border):
        for j in range(to_border, len(img[0]) - to_border):
            for k in range(len(img[0,0])):
                new_px = 0
                for x in range(-to_border, to_border+1):
                    for y in range(-to_border, to_border+1):
                        new_px += filter[x+to_border,y+to_border]*img[i+x, j+y, k]
                result[i, j, k] = new_px
    return result

def apply_median_filter(img, size=3):
    if size%2 == 0:
        print('Tiene que ser impar')
        return img
    to_border = int((size-1)/2)
    result = img.copy()
    for i in range(to_border, len(img)-to_border):
        for j in range(to_border, len(img[0]) - to_border):
            for k in range(len(img[0,0])):
                new_px = []
                for x in range(-to_border, to_border+1):
                    for y in range(-to_border, to_border+1):
                        new_px.append(img[i+x, j+y, k])
                new_px = np.sort(new_px)
                result[i, j, k] = new_px[int((len(new_px)-1)/2)]
    return result

def apply_weighted_median_filter(img):
    to_border = 1
    filter = [[1,2,1], [2,4,2], [1,2,1]]
    result = img.copy()
    for i in range(to_border, len(img)-to_border):
        for j in range(to_border, len(img[0]) - to_border):
            for k in range(len(img[0,0])):
                new_px = []
                for x in range(-to_border, to_border+1):
                    for y in range(-to_border, to_border+1):
                        for _ in range(filter[x+to_border][y+to_border]):
                            new_px.append(img[i+x, j+y, k])
                new_px = np.sort(new_px)
                result[i, j, k] = new_px[int((len(new_px)-1)/2)]
    return result

def apply_gauss_filter(img, sigma=1):
    size = 2*sigma+1
    filter = np.ones(size * size).reshape((size, size)) / (size * size)
    to_border = int((size - 1) / 2)

    for x in range(-to_border, to_border + 1):
        for y in range(-to_border, to_border + 1):
            filter[to_border+x][to_border+y] = (1/math.sqrt(2*math.pi*sigma*sigma))*math.exp(-(x*x+y*y)/(sigma*sigma))
    filter /= np.sum(filter)

    result = img.copy()
    for i in range(to_border, len(img) - to_border):
        for j in range(to_border, len(img[0]) - to_border):
            for k in range(len(img[0, 0])):
                new_px = 0
                for x in range(-to_border, to_border + 1):
                    for y in range(-to_border, to_border + 1):
                        new_px += filter[x + to_border, y + to_border] * img[i + x, j + y, k]
                result[i, j, k] = new_px
    return result

def apply_pasaalto_filter(img):
    to_border = 1
    filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 9
    result = img.copy()
    for i in range(to_border, len(img) - to_border):
        for j in range(to_border, len(img[0]) - to_border):
            for k in range(len(img[0, 0])):
                new_px = 0
                for x in range(-to_border, to_border + 1):
                    for y in range(-to_border, to_border + 1):
                        new_px += filter[x + to_border, y + to_border] * img[i + x, j + y, k]
                result[i, j, k] = new_px
    return result

def _apply_between_images(f, img1, img2):
    width = max(len(img1), len(img2))
    height = max(len(img1[0]), len(img2[0]))
    result = np.zeros((width, height, 3), np.int32)
    img1 = tr.complete_with_zeros(img1, width, height)
    img2 = tr.complete_with_zeros(img2, width, height)
    extremeValues = [0, 0]
    for i in range(width):
        for j in range(height):
            for k in range(len(img1[i,j])):
                result[i,j,k] = f(np.int32(img1[i,j,k]), np.int32(img2[i,j,k]))
                extremeValues[1] = max(result[i,j,k], extremeValues[1])
                extremeValues[0] = min(result[i,j,k], extremeValues[0])
    result = tr.mapValues(result, extremeValues[0], extremeValues[1])
    return result

def _cdf(p, n):
    result = np.copy(p)
    # initialize
    result[0] = p[0] / n
    for x in range(len(p) - 1):
        for y in range(len(p[0])):
            result[x + 1, y] = p[x + 1, y] / n + result[x, y]
    return result

def _hist(img, width, height):
    colorsCount = np.zeros((256, 3))
    for i in range(width):
        for j in range(height):
            for k in range(len(img[i,j])):
                colorsCount[img[i,j,k], k] = colorsCount[img[i,j,k], k] + 1
    return colorsCount

def _cdf_min(cdf):
    minCdf = np.zeros(3)
    for x in range(len(cdf)):
        for y in range(len(cdf[0])):
            minCdf[y] = min (minCdf[y], cdf[x, y])
    return minCdf
