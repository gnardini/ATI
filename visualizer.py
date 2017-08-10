import cv2
import numpy as np

def stack_images(images):
    return np.hstack(images)

def show_image(image):
    cv2.imshow('ATI', image)
    cv2.waitKey(0)

