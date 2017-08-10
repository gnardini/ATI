import cv2
import numpy as np

def stack_images(images):
    return np.hstack(images)

def show_image(image, window='ATI'):
    cv2.imshow(window, image)
    cv2.waitKey(0)

def bindMouseCallbacks(fx, window='ATI'):
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, fx)