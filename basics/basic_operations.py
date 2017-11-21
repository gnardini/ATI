import cv2
import numpy as np

# Ang should be in grades
def rotate(img, ang = 90):
    rows,cols,c = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
    return cv2.warpAffine(img,M,(cols,rows))

def scale(img, value=2):
    return cv2.resize(img,None,fx=value, fy=value, interpolation = cv2.INTER_CUBIC)

def translate(img, x=100, y=20):
    rows,cols,c = img.shape
    M = np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(img,M,(cols,rows))

def perspective(img):
    rows,cols,c = img.shape
    pts1 = np.float32([[10,0],[60,30],[28,50],[90,20]])
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,(cols,rows))
