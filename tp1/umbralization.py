import numpy as np
import math
from tp1 import image_operations as ops
from basics import transforms as tr

def global_thresholding(img, delta_t=3):
    thresh = 128
    T_diff = delta_t + 1
    th = img
    while T_diff > delta_t:
        th = ops.apply_threshold(img, thresh)
        m1 = 0
        m2 = 0
        m1_total = 0
        m2_total = 0
        for i in range(len(img)):
            for j in range(len(img[i])):
                if th[i, j, 0] == 0:
                    m1 += img[i, j, 0]
                    m1_total += 1
                else:
                    m2 += img[i, j, 0]
                    m2_total += 1
        m1 /= m1_total
        m2 /= m2_total
        new_t = (m1 + m2) / 2
        T_diff = abs(thresh - new_t)
        thresh = new_t
    return th

def otsu(img):
    hist = ops.grayscale_histogram(img, False)
    acum = np.cumsum(hist)
    means = np.zeros(256)
    for i in range(1, 256):
        means[i] = means[i-1] + acum[i] * i
    mg = means[255]
    variances = np.zeros(256)
    for i in range(256):
        m1 = means[i] / acum[i]
        m2 = (mg - means[i]) / (1 - acum[i])
        vari = acum[i] * (1 - acum[i])
        vari *= (m1-m2)**2
        print(i, m1, m2, acum[i] * (1 - acum[i]), (m1-m2)**2)
        variances[i] = vari
        # mF = (mg - means[i]) / (1 - acum[i])
        # variances[i] = (means[i] / acum[i] - mF) ** 2 * (acum[i] * (1 - acum[i]))
        # variances[i] = (mg * acum[i] - means[i])**2 * (acum[i] * (1 - acum[i]))
    print(variances)
    threshold = np.argmax(variances)
    print(np.argmax(variances))
    return ops.apply_threshold(img, threshold)