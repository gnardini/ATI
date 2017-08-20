import __init__
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import image_io
import visualizer
import cv2

from tp1 import image_operations as ops

IMG_FILE = '../images/TEST.PGM'
img = image_io.read(IMG_FILE)

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
originalPanelA.grid(row=base_row, column=0, columnspan=2)
resultPanelA = Label(root)
resultPanelA.grid(row=base_row, column=2, columnspan=2)
btnA = Button(root, text="Elegir imagen", command=assign_image_a)
btnA.grid(row=base_row+1, column=0)
originalPanelB = Label(root)
originalPanelB.grid(row=base_row+2, column=0, columnspan=2)
btnB = Button(root, text="Elegir imagen", command=assign_image_b)
btnB.grid(row=base_row+3, column=0)
resultPanelB = Label(root)
resultPanelB.grid(row=base_row+2, column=2, columnspan=2)

# Botones de transformacion
add_btn = Button(root, text='Sumar', command=lambda: put_into(resultPanelA, ops.add_images(original_image_a, original_image_b)))
add_btn.grid(row=0, column=0)
sub_btn = Button(root, text='Restar', command=lambda: put_into(resultPanelA, ops.subtract_images(original_image_a, original_image_b)))
sub_btn.grid(row=0, column=1)
mult_btn = Button(root, text='Multiplicar', command=lambda: put_into(resultPanelA, ops.multiply_images(original_image_a, original_image_b)))
mult_btn.grid(row=0, column=2)
mult_btn = Button(root, text='Histograma', command=lambda: ops.grayscale_histogram(original_image_a))
mult_btn.grid(row=0, column=3)
negative_btn = Button(root, text='Negativo', command=lambda: put_into(resultPanelA, ops.negative(original_image_a)))
negative_btn.grid(row=0, column=4)
negative_btn = Button(root, text='Contraste', command=lambda: put_into(resultPanelA, ops.increase_contrast(original_image_a)))
negative_btn.grid(row=0, column=5)

root.mainloop()
