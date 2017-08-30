import numpy as np

def map_dynamic_value(pixel, const):
    if (pixel <= 0):
        return np.uint8(pixel)
    return np.uint8(const*np.log(1 + pixel))

def map_value(pixel, minValue, maxValue):
    range = (maxValue - minValue)
    if (range == 0):
        return pixel
    else:
        return np.uint8((pixel - minValue) * 255 / range)

def sum(pixel1, pixel2):
    return pixel1 + pixel2

def subtract(pixel1, pixel2):
    return pixel1 - pixel2

def multiply(pixel1, pixel2):
    return pixel1 * pixel2
