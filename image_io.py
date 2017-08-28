import io
import cv2
import numpy as np

sizes = {
    'GIRL.RAW': (389, 164),
    'BARCO.RAW': (290, 207),
    'LENA.RAW': (256, 256),
    'GIRL2.RAW': (256, 256),
    'FRACTAL.RAW': (200, 200),
}

def read(file_path, grayscale=False):
    file_name = file_path.split('/')[-1]
    if file_name in sizes:
        return _read_raw(file_path, sizes[file_name][0], sizes[file_name][1])
    else:
        return cv2.imread(file_path)

def _read_raw(file, width, height):
    image = np.zeros((height, width, 3), np.uint8)
    with io.open(file, 'rb') as f:
        text = f.read()
        for k in range(len(text)):
            image[k // width][k % width][0] = text[k]
            image[k // width][k % width][1] = text[k]
            image[k // width][k % width][2] = text[k]
    return image

def _read_img(file, grayscale):
    img = cv2.imread(file)
    if grayscale:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img

def save_image(image, path):
    cv2.imwrite(path, image)