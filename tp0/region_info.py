import __init__
import image_io
import visualizer
import cv2

IMG_FILE = '../images/TEST.PGM'
img = image_io.read(IMG_FILE)

choosing = False
start = (0, 0)

def click(event, x, y, flags, param):
    global choosing
    if event == cv2.EVENT_LBUTTONDOWN:
        choosing = True
        global start
        start = (x, y)
    elif choosing and event == cv2.EVENT_LBUTTONUP:
        choosing = False
        minX, maxX = (min(start[0], x), max(start[0], x))
        minY, maxY = (min(start[1], y), max(start[1], y))
        total = (maxX - minX) * (maxY - minY)
        if (len(img[0][0]) == 0):
            totalValue = [0]
        else:
            totalValue = [0, 0, 0]
        for i in range(maxX-minX):
            for j in range(maxY-minY):
                pixel = img[minY+j][minX+i]
                for k in range(len(pixel)):
                    totalValue[k] += pixel[k]
        for k in range(len(totalValue)):
            totalValue[k] /= total
        print('Total px: %d. Avg value: %s' % (total, totalValue))

visualizer.bindMouseCallbacks(click)
visualizer.show_image(img)
