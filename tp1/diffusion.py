import numpy as np
import math
from tp1 import image_operations as ops
from basics import transforms as tr

def isotropic_diffusion(img, t):
    size = 2 * t + 1
    filter = np.zeros(size * size).reshape((size, size))
    to_border = int((size - 1) / 2)

    for x in range(-to_border, to_border + 1):
        for y in range(-to_border, to_border + 1):
            filter[to_border + x][to_border + y] = (1 / (4 * math.pi * t)) * math.exp(
                -(x * x + y * y) / (4 * t))
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

def leclerc(gradient, sigma):
    return math.exp(-math.pow(gradient, 2) / math.pow(sigma, 2))

def lorentz(gradient, sigma):
    return 1 / (1 + math.pow(gradient, 2) / math.pow(sigma, 2))

def anisotropic_diffusion(img, t, sigma, function):
    while t > 0:
        result = np.zeros(img.shape)
        for i in range(1, len(img)-1):
            for j in range(1, len(img[i])-1):
                for k in range(len(img[i, j])):
                    new_px = img[i,j,k]
                    gradients = [
                        img[i + 1, j, k] - img[i, j, k], # N
                        img[i - 1, j, k] - img[i, j, k], # S
                        img[i, j + 1, k] - img[i, j, k], # E
                        img[i, j - 1, k] - img[i, j, k] # O
                    ]
                    if function == 1:
                        g = [gradient * leclerc(gradient, sigma) for gradient in gradients]
                    else:
                        g = [gradient * lorentz(gradient, sigma) for gradient in gradients]
                    new_px += np.sum(g) * .25
                    result[i,j,k] = new_px
        img = result
        t -= 1

    return tr.mapValues(img, np.min(img), np.max(img))
