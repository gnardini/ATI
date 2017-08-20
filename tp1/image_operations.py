import cv2
import numpy as np

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

def negative(img):
    result = np.copy(img)
    return np.vectorize(lambda x: 255 - x)(result).astype(np.uint8)

def grayscale_histogram(img, show_plot=True):
    colorsCount = np.zeros(256)
    width = len(img)
    height = len(img[0])
    for i in range(width):
        for j in range(height):
            for k in range(len(img[i,j])):
                colorsCount[img[i,j,k]] += 1
    for x in range(len(colorsCount)):
        colorsCount[x] /= (width * height * len(img[0,0]))
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

def equalize(img):
    print('TODO')

def random_gauss(mean, stdv):
    print('TODO')

def random_rayleigh(epsilon):
    print('TODO')

def random_exponential(lambda_):
    print('TODO')

def ejercicio_9_para_adelante():
    print('TODO')

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
