import cv2
import numpy as np

def add_images(img1, img2):
    print('TODO')

def subtract_images(img1, img2):
    print('TODO')

def multiply_images(img1, img2):
    print('TODO')

def negative(img):
    result = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i,j])):
                result[i,j,k] = 255 - img[i,j,k]
    return result

def grayscale_histogram(img):
    print('TODO')

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
    print('TODO')

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

