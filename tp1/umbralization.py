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
    width = len(img)
    height = len(img[0])
    hist = ops._hist(img, len(img), len(img[0]))
    cdf = ops._cdf(hist, len(img) * len(img[0]))
    print(cdf)
    total = width * height
    current_max, threshold = 0, 0
    sumT, sumF, sumB = 0, 0, 0
    for i in range(0,256):
        sumT += i * hist[i][0]
        weightB, weightF = 0, 0
        varBetween, meanB, meanF = 0, 0, 0
    for i in range(0,256):
        weightB += hist[i][0]
        weightF = total - weightB
        if weightF == 0:
            break
        sumB += i*hist[i][0]
        sumF = sumT - sumB
        meanB = sumB/weightB
        meanF = sumF/weightF
        varBetween = weightB * weightF
        varBetween *= (meanB-meanF)*(meanB-meanF)
        if varBetween > current_max:
            current_max = varBetween
            threshold = i
    print (threshold)
    return ops.apply_threshold(img, threshold)
