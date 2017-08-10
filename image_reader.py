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

def read(file_path):
    file_name = file_path.split('/')[-1]
    if file_name in sizes:
        return _read_raw(file_path, sizes[file_name][0], sizes[file_name][1])
    else:
        return cv2.imread(file_path)

def _read_raw(file, width, height):
    image = np.zeros((height, width, 1), np.uint8)
    with io.open(file, 'r', encoding='mac_roman') as f:
        text = f.read()
        for k in range(len(text)):
            image[k // width][k % width] = ord(text[k])
    return image
