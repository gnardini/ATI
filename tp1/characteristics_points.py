import numpy as np
import cv2
import math
from tp1 import image_operations as ops
from tp1 import umbralization as umb
from basics import transforms as tr
from tp1 import border_detection as bd

def harris(img, threshold = 0, percentage = 0.05, k = 0.04):
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gs_img_3 = np.zeros_like(img)
    gs_img_3[:,:,0] = gs_img
    gs_img_3[:,:,1] = gs_img
    gs_img_3[:,:,2] = gs_img

    ## Apply prewitt masks
    vertical_mask = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    horizontal_mask = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    vertical = ops.apply_mask(gs_img_3, vertical_mask)
    horizontal = ops.apply_mask(gs_img_3, horizontal_mask)

    ## Calculate components from masks
    vertical_2 = vertical * vertical
    horizontal_2 = horizontal * horizontal
    vertical_horizontal = vertical * vertical

    ## Apply Gauss filter
    vertical_2 = ops.apply_gauss_filter(vertical_2, 2)
    horizontal_2 = ops.apply_gauss_filter(horizontal_2, 2)
    vertical_horizontal = ops.apply_gauss_filter(vertical_horizontal, 2)

    part1 = vertical_horizontal * vertical_horizontal
    part2 = vertical_2 + horizontal_2

    cim =(horizontal_2 * vertical_2 - part1 - (part2 * part2) * k)
    maxCim = cim.max();

    if threshold == 0:
        threshold = maxCim*percentage

    print("Threshold: %d"%(threshold))
    # Add result to original image
    result = np.copy(img)
    for i in range(len(result)):
        for j in range(len(result[i])):
            # 100000000
            if (cim[i,j,0] > threshold):
                result[i,j,0] = 0
                result[i,j,1] = 0
                result[i,j,2] = 255
    return result

def sift_comparison(img1, img2, percentage=0.75):
    # create BFMatcher object
    bf = cv2.BFMatcher(crossCheck=True)

    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray,None)
    result1 = cv2.drawKeypoints(gray,kp1,img1)

    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift.detectAndCompute(gray,None)
    result2 = cv2.drawKeypoints(gray,kp2,img2)

    # Match descriptors.
    matches = bf.match(des1, des2)
    if (len(matches) >= percentage * len(des1)) || (len(matches) >= percentage * len(des2)):
        print('Match')
    else:
        print('Not match')

    return [result1, result2]
# Para sift. Contar cantidad de descriptores en la imagen original, la cantidad de descriptores en la imagen a comparar
# Armar relación entre descriptores y coincidencias. (contar las coincidencias de descriptores)
# Después de borronear tanto lo que queda es realmente característico de la imagen
# Octavas

# Los frameworks suele determinar cuántos sigmas se usan.
# Curvatura principales de una superficie es ??
#   Curvatura principal -> lambda 1 * lambda2 / (lambda 1 + lambda2)
#   Deja el análisis de harris dependiente del radio de curvatura
#   Hay que verificar si el ru = 10 está bien, se fija que los autovalores no sean 10 veces más grandes
