import __init__
from tkinter import *
from tkinter import filedialog
import numpy as np
import image_io
import time

from tp1 import image_operations as ops
from tp1 import border_detection as bd
from tp1 import umbralization as umb
from tp1 import diffusion as diff
from tp1 import active_contours as ac
from tp1 import motion_handler as mh
from tp1 import image_management as im

# img_format = './images/video sintetico/a{}.jpg'
img_format = './images/video1/Pic{}.jpg'

class VideoActiveContour:

    def __init__(self, root, label):
        self.manager = im.ImageManagement()
        self.root = root
        self.label = label
        self.motion_handler = mh.MotionHandler(self.manager)
        self.current_frame = 1
        self.blocked = False

    def read_next_img(self):
        path = img_format.format(self.current_frame)
        img = image_io.read(path)
        self.manager.put_into('original-up', img)

    def load_video(self):
        self.current_frame = 1
        self.blocked = False
        self.manager.clear_cache()
        self.read_next_img()
        self.motion_handler.start()

    def adjust_contours(self):
        self.manager.restore_original()
        img = self.manager.get_image('original-up')
        if self.current_frame == 1:
            self.last_frame = time.clock()
            img, contours, lin, lout = ac.active_contours_rect(img, self.motion_handler.get_rect())
        else:
            current_time = time.clock()
            diff = current_time - self.last_frame
            time_text = 'Tiempo (segs): %f' % diff
            self.label.set(time_text)
            self.last_frame = current_time
            img, contours, lin, lout = ac.active_contours(img, self.contours, self.lin, self.lout, 10)
        self.current_frame += 1
        self.manager.put_into('original-up', img)
        self.contours = contours
        self.lin = lin
        self.lout = lout
        self.blocked = True
        self.root.update()
        self.next_image()

    def next_image(self):
        self.blocked = False
        self.read_next_img()
        self.manager.save_original()
        img = self.manager.get_image('original-up')
        ac.draw_contours(img, self.lin, self.lout)
        self.manager.put_into('original-up', img)
        self.adjust_contours()

root = Tk()

label_text = StringVar()
label = Label(root, textvariable=label_text, font="Default 16 bold")
label.grid(row=2, column=0, columnspan=3)

active = VideoActiveContour(root, label_text)

btn = Button(root, text='Empezar', command=active.load_video)
btn.grid(row=0, column=0)
btn = Button(root, text='Ajustar contornos', command=active.adjust_contours)
btn.grid(row=0, column=1)
btn = Button(root, text='Siguiente imagen', command=active.next_image)
btn.grid(row=0, column=2)

active.manager.set_panel('original-up', Label(root))
active.manager.get_panel('original-up').grid(row=1, column=0, columnspan=3)

root.mainloop()
