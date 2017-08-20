import __init__
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import numpy as np
import image_io

from tp1 import image_operations as ops

def to_tk_image(img):
    return ImageTk.PhotoImage(Image.fromarray(img))

def from_tk_image(img):
    return

def select_image(targetPanel):
    path = filedialog.askopenfilename()
    if len(path) > 0:
        img_array = image_io.read(path)
        img = to_tk_image(img_array)
        targetPanel.configure(image=img)
        targetPanel.image = img
    else:
        print('Invalid image')
        return np.zeros(0)
    return img_array

def put_into(panel, img):
    img = to_tk_image(img)
    panel.configure(image=img)
    panel.image = img

def assign_image_a():
    global original_image_a
    original_image_a = select_image(originalPanelA)

def assign_image_b():
    global original_image_b
    original_image_b = select_image(originalPanelB)

root = Tk()

original_image_a = None
original_image_b = None

# Imagenes y botones para agregarlas
base_row = 2
originalPanelA = Label(root)
originalPanelA.grid(row=base_row, column=0, columnspan=3)
resultPanelA = Label(root)
resultPanelA.grid(row=base_row, column=3, columnspan=3)
btnA = Button(root, text="Elegir imagen", command=assign_image_a)
btnA.grid(row=base_row+1, column=0)
originalPanelB = Label(root)
originalPanelB.grid(row=base_row+2, column=0, columnspan=3)
btnB = Button(root, text="Elegir imagen", command=assign_image_b)
btnB.grid(row=base_row+3, column=0)
resultPanelB = Label(root)
resultPanelB.grid(row=base_row+2, column=3, columnspan=3)

# Botones de transformacion
btn = Button(root, text='Sumar', command=lambda: put_into(resultPanelA, ops.add_images(original_image_a, original_image_b)))
btn.grid(row=0, column=0)
btn = Button(root, text='Restar', command=lambda: put_into(resultPanelA, ops.subtract_images(original_image_a, original_image_b)))
btn.grid(row=0, column=1)
btn = Button(root, text='Multiplicar', command=lambda: put_into(resultPanelA, ops.multiply_images(original_image_a, original_image_b)))
btn.grid(row=0, column=2)
btn = Button(root, text='Histograma', command=lambda: ops.grayscale_histogram(original_image_a))
btn.grid(row=0, column=3)
btn = Button(root, text='Negativo', command=lambda: put_into(resultPanelA, ops.negative(original_image_a)))
btn.grid(row=0, column=4)
btn = Button(root, text='Contraste', command=lambda: put_into(resultPanelA, ops.increase_contrast(original_image_a)))
btn.grid(row=0, column=5)
scale = Scale(root, from_=0, to=255, orient=HORIZONTAL)
scale.set(128)
scale.grid(row=0, column=6)
btn = Button(root, text='Umbralización', command=lambda: put_into(resultPanelA, ops.apply_threshold(original_image_a, scale.get())))
btn.grid(row=1, column=6)

root.mainloop()
