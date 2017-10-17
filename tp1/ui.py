import __init__
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import numpy as np
import image_io

from tp1 import image_operations as ops
from tp1 import border_detection as bd
from tp1 import umbralization as umb
from tp1 import diffusion as diff

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

def assign_image(key):
    images[key] = select_image(panels[key])

#TODO: fix this function
def save_image(key):
    path = filedialog.asksaveasfilename()
    if path:
        image_io.save_image(images[key], path)

def put_into(key, img):
    images[key] = img
    img = to_tk_image(img)
    panels[key].configure(image=img)
    panels[key].image = img

root = Tk()

panels = {
    'original-up': None,
    'original-down': None,
    'result-up': None,
    'result-down': None
}
images = {
    'original-up': None,
    'original-down': None,
    'result-up': None,
    'result-down': None
}

# Imagenes y botones para agregarlas
base_row = 9
panels['original-up'] = Label(root)
panels['original-up'].grid(row=base_row, column=0, columnspan=3)
panels['result-up'] = Label(root)
panels['result-up'].grid(row=base_row, column=3, columnspan=3)
btnA = Button(root, text="Elegir imagen", command=lambda: assign_image('original-up'))
btnA.grid(row=base_row+1, column=0)
btnA = Button(root, text="Mover abajo", command=lambda: put_into('original-down', images['original-up']))
btnA.grid(row=base_row+1, column=1)
btnA = Button(root, text="Guardar imagen", command=lambda: save_image('result-up'))
btnA.grid(row=base_row+1, column=3)
btnA = Button(root, text="Mover a izquierda", command=lambda: put_into('original-up', images['result-up']))
btnA.grid(row=base_row+1, column=4)
panels['original-down'] = Label(root)
panels['original-down'].grid(row=base_row+2, column=0, columnspan=3)
btnB = Button(root, text="Elegir imagen", command=lambda: assign_image('original-down'))
btnB.grid(row=base_row+3, column=0)
panels['result-down'] = Label(root)
panels['result-down'].grid(row=base_row+2, column=3, columnspan=3)

# Botones de transformacion
parameter = Scale(root, from_=0, to=10, resolution=0.1, orient=HORIZONTAL)
parameter.set(1)
parameter.grid(row=2, column=1)

btn = Button(root, text='Sumar', command=lambda: put_into('result-up', ops.add_images(images['original-up'], images['original-down'])))
btn.grid(row=0, column=0)
btn = Button(root, text='Restar', command=lambda: put_into('result-up', ops.subtract_images(images['original-up'], images['original-down'])))
btn.grid(row=0, column=1)
btn = Button(root, text='Multiplicar', command=lambda: put_into('result-up', ops.multiply_images(images['original-up'], images['original-down'])))
btn.grid(row=1, column=0)
btn = Button(root, text='Negativo', command=lambda: put_into('result-up', ops.negative(images['original-up'])))
btn.grid(row=1, column=1)
btn = Button(root, text='Histograma', command=lambda: ops.grayscale_histogram(images['original-up']))
btn.grid(row=0, column=2)
btn = Button(root, text='Ecualización', command=lambda: put_into('result-up', ops.equalize(images['original-up'])))
btn.grid(row=1, column=2)
btn = Button(root, text='Contraste', command=lambda: put_into('result-up', ops.increase_contrast(images['original-up'])))
btn.grid(row=0, column=4)
scalarScale = Scale(root, from_=1.5, to=10, resolution=0.5, orient=HORIZONTAL)
scalarScale.set(1.5)
scalarScale.grid(row=0, column=5)
btn = Button(root, text='Multiplica constante', command=lambda: put_into('result-up', ops.multiply_by_scalar(images['original-up'], scalarScale.get())))
btn.grid(row=1, column=5)
scale = Scale(root, from_=0, to=255, orient=HORIZONTAL)
scale.set(128)
scale.grid(row=0, column=3)
btn = Button(root, text='Umbralización', command=lambda: put_into('result-up', ops.apply_threshold(images['original-up'], scale.get())))
btn.grid(row=1, column=3)
btn = Button(root, text='Filtro Gamma', command=lambda: put_into('result-up', ops.apply_gamma_potential(images['original-up'], parameter.get())))
btn.grid(row=1, column=4)
gaussScale = Scale(root, from_=0, to=1, resolution=0.01, orient=HORIZONTAL)
gaussScale.set(.2)
gaussScale.grid(row=2, column=0)
btn = Button(root, text='Gauss', command=lambda: put_into('result-up', ops.add_gaussian_noise(images['original-up'], gaussScale.get(), parameter.get())))
btn.grid(row=2, column=2)
btn = Button(root, text='Exponencial', command=lambda: put_into('result-up', ops.add_exponential_noise(images['original-up'], gaussScale.get(), parameter.get())))
btn.grid(row=2, column=3)
btn = Button(root, text='Rayleigh', command=lambda: put_into('result-up', ops.add_rayleigh_noise(images['original-up'], gaussScale.get(), parameter.get())))
btn.grid(row=2, column=4)
btn = Button(root, text='Sal y Pimienta', command=lambda: put_into('result-up', ops.add_salt_pepper_noise(images['original-up'], gaussScale.get())))
btn.grid(row=2, column=5)
filterScale = Scale(root, from_=2, to=15, orient=HORIZONTAL)
filterScale.set(5)
filterScale.grid(row=3, column=0)
btn = Button(root, text='Filtro media', command=lambda: put_into('result-up', ops.apply_mean_filter(images['original-up'], filterScale.get())))
btn.grid(row=3, column=1)
btn = Button(root, text='Filtro mediana', command=lambda: put_into('result-up', ops.apply_median_filter(images['original-up'], filterScale.get())))
btn.grid(row=3, column=2)
btn = Button(root, text='Filtro mediana ponderado', command=lambda: put_into('result-up', ops.apply_weighted_median_filter(images['original-up'])))
btn.grid(row=3, column=3)
btn = Button(root, text='Filtro Gauss', command=lambda: put_into('result-up', ops.apply_gauss_filter(images['original-up'])))
btn.grid(row=3, column=4)
btn = Button(root, text='Filtro pasaalto', command=lambda: put_into('result-up', ops.apply_pasaalto_filter(images['original-up'])))
btn.grid(row=3, column=5)
btn = Button(root, text='Prewitt', command=lambda: put_into('result-up', bd.prewitt(images['original-up'])))
btn.grid(row=4, column=0)
btn = Button(root, text='Sobel', command=lambda: put_into('result-up', bd.sobel(images['original-up'])))
btn.grid(row=4, column=1)
btn = Button(root, text='Prewitt directional', command=lambda: put_into('result-up', bd.directional_prewitt(images['original-up'])))
btn.grid(row=4, column=2)
btn = Button(root, text='Sobel directional', command=lambda: put_into('result-up', bd.directional_sobel(images['original-up'])))
btn.grid(row=4, column=3)
btn = Button(root, text='Laplaciano', command=lambda: put_into('result-up', bd.laplace(images['original-up'])))
btn.grid(row=5, column=0)
laplaceScale = Scale(root, from_=0, to=255, orient=HORIZONTAL)
laplaceScale.set(128)
laplaceScale.grid(row=5, column=1)
btn = Button(root, text='Laplaciano pendiente', command=lambda: put_into('result-up', bd.laplace_pendiente(images['original-up'], laplaceScale.get())))
btn.grid(row=5, column=2)
sigmaScale = Scale(root, from_=1, to=11, orient=HORIZONTAL)
sigmaScale.set(1)
sigmaScale.grid(row=5, column=3)
btn = Button(root, text='Laplaciano gauss', command=lambda: put_into('result-up', bd.laplace_gauss(images['original-up'], sigmaScale.get(), laplaceScale.get())))
btn.grid(row=5, column=4)
deltaTScale = Scale(root, from_=1, to=15, orient=HORIZONTAL)
deltaTScale.set(3)
deltaTScale.grid(row=6, column=0)
btn = Button(root, text='Umbralización global', command=lambda: put_into('result-up', umb.global_thresholding(images['original-up'], deltaTScale.get())))
btn.grid(row=6, column=1)
btn = Button(root, text='Otsu', command=lambda: put_into('result-up', umb.otsu(images['original-up'])))
btn.grid(row=6, column=2)
tScale = Scale(root, from_=1, to=5, orient=HORIZONTAL)
tScale.set(3)
tScale.grid(row=7, column=0)
sigScale = Scale(root, from_=20, to=100, resolution=5, orient=HORIZONTAL)
sigScale.set(20)
sigScale.grid(row=7, column=1)
btn = Button(root, text='Difusion Isotropica', command=lambda: put_into('result-up', diff.isotropic_diffusion(images['original-up'], tScale.get())))
btn.grid(row=7, column=2)
btn = Button(root, text='Difusion Anisotropica', command=lambda: put_into('result-up', diff.anisotropic_diffusion(images['original-up'], tScale.get(), sigScale.get(), funcSel.get())))
btn.grid(row=7, column=3)
funcSel = Scale(root, from_=1, to=2, orient=HORIZONTAL)
funcSel.set(1)
funcSel.grid(row=7, column=4)
btn = Button(root, text='Canny', command=lambda: put_into('result-up', bd.canny_detector(images['original-up'])))
btn.grid(row=8, column=0)
btn = Button(root, text='SUSAN', command=lambda: put_into('result-up', bd.susan(images['original-up'])))
btn.grid(row=8, column=1)

root.mainloop()
