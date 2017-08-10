import __init__
import numpy as np
import cv2
import image_io

image = np.zeros((200, 200, 1), np.uint8)
cv2.circle(image, (100, 100), 40, (255, 255, 255), -1)
image_io.save_image(image, '../images/circle.png')
