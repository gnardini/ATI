import __init__
from tkinter import *
from tkinter import filedialog
import numpy as np
import image_io

from tp1 import image_operations as ops
from tp1 import border_detection as bd
from tp1 import umbralization as umb
from tp1 import diffusion as diff
from tp1 import active_contours as ac
from tp1 import motion_handler as mh
from tp1 import image_management as im

def from_tk_image(img):
    return

def read_image_to(target_panel, path):
    img_array = image_io.read(path)
    img = manager.to_tk_image(img_array)
    target_panel.configure(image=img)
    target_panel.image = img
    return img_array

def select_image(targetPanel):
    path = filedialog.askopenfilename()
    if len(path) > 0:
        return read_image_to(targetPanel, path)
    else:
        print('Invalid image')
        return np.zeros(0)

def assign_image(key):
    manager.clear_cache()
    manager.set_image(key, select_image(manager.get_panel(key)))

def save_image(key):
    path = filedialog.asksaveasfilename()
    if path:
        image_io.save_image(manager.get_image(key), path)

def put_into(key, img):
    manager.put_into(key, img)

def start_choose_region():
    motion_handler.start()

root = Tk()
buttons = Toplevel(root)

manager = im.ImageManagement()
motion_handler = mh.MotionHandler(manager)

# Imagenes y botones para agregarlas
base_row = 0
manager.set_panel('original-up', Label(buttons))
manager.get_panel('original-up').grid(row=base_row, column=0, columnspan=3)
manager.set_panel('result-up', Label(buttons))
manager.get_panel('result-up').grid(row=base_row, column=3, columnspan=3)
btnA = Button(buttons, text="Elegir imagen", command=lambda: assign_image('original-up'))
btnA.grid(row=base_row+1, column=0)
btnA = Button(buttons, text="Mover abajo", command=lambda: put_into('original-down', manager.get_image('original-up')))
btnA.grid(row=base_row+1, column=1)
btnA = Button(buttons, text="Guardar imagen", command=lambda: save_image('result-up'))
btnA.grid(row=base_row+1, column=3)
btnA = Button(buttons, text="Mover a izquierda", command=lambda: put_into('original-up', manager.get_image('result-up')))
btnA.grid(row=base_row+1, column=4)
manager.set_panel('original-down', Label(buttons))
manager.get_panel('original-down').grid(row=base_row+2, column=0, columnspan=3)
btnB = Button(buttons, text="Elegir imagen", command=lambda: assign_image('original-down'))
btnB.grid(row=base_row+3, column=0)
btn = Button(buttons, text="Mover arriba", command=lambda: put_into('original-up', manager.get_image('original-down')))
btn.grid(row=base_row+3, column=1)
manager.set_panel('result-down', Label(buttons))
manager.get_panel('result-down').grid(row=base_row+2, column=3, columnspan=3)

manager.set_image('original-up', read_image_to(manager.get_panel('original-up'), './images/LENA.RAW'))

# Botones de transformacion

btn = Button(root, text='Sumar', command=lambda: put_into('result-up', ops.add_images(manager.get_image('original-up'), manager.get_image('original-down'))))
btn.grid(row=0, column=0)
btn = Button(root, text='Restar', command=lambda: put_into('result-up', ops.subtract_images(manager.get_image('original-up'), manager.get_image('original-down'))))
btn.grid(row=0, column=1)
btn = Button(root, text='Multiplicar', command=lambda: put_into('result-up', ops.multiply_images(manager.get_image('original-up'), manager.get_image('original-down'))))
btn.grid(row=1, column=0)
btn = Button(root, text='Negativo', command=lambda: put_into('result-up', ops.negative(manager.get_image('original-up'))))
btn.grid(row=1, column=1)
btn = Button(root, text='Histograma', command=lambda: ops.grayscale_histogram(manager.get_image('original-up')))
btn.grid(row=0, column=2)
btn = Button(root, text='Ecualización', command=lambda: put_into('result-up', ops.equalize(manager.get_image('original-up'))))
btn.grid(row=1, column=2)
btn = Button(root, text='Contraste', command=lambda: put_into('result-up', ops.increase_contrast(manager.get_image('original-up'))))
btn.grid(row=0, column=4)
scalarScale = Scale(root, from_=1.5, to=10, resolution=0.5, orient=HORIZONTAL)
scalarScale.set(1.5)
scalarScale.grid(row=0, column=5)
btn = Button(root, text='Multiplica constante', command=lambda: put_into('result-up', ops.multiply_by_scalar(manager.get_image('original-up'), scalarScale.get())))
btn.grid(row=1, column=5)
scale = Scale(root, from_=0, to=255, orient=HORIZONTAL)
scale.set(128)
scale.grid(row=0, column=3)
btn = Button(root, text='Umbralización', command=lambda: put_into('result-up', ops.apply_threshold(manager.get_image('original-up'), scale.get())))
btn.grid(row=1, column=3)
btn = Button(root, text='Filtro Gamma', command=lambda: put_into('result-up', ops.apply_gamma_potential(manager.get_image('original-up'), parameter.get())))
btn.grid(row=1, column=4)
gaussScale = Scale(root, from_=0, to=1, resolution=0.01, orient=HORIZONTAL)
gaussScale.set(.2)
gaussScale.grid(row=2, column=0)
parameter = Scale(root, from_=0, to=10, resolution=0.1, orient=HORIZONTAL)
parameter.set(1)
parameter.grid(row=2, column=1)
btn = Button(root, text='Gauss', command=lambda: put_into('result-up', ops.add_gaussian_noise(manager.get_image('original-up'), gaussScale.get(), parameter.get())))
btn.grid(row=2, column=2)
btn = Button(root, text='Exponencial', command=lambda: put_into('result-up', ops.add_exponential_noise(manager.get_image('original-up'), gaussScale.get(), parameter.get())))
btn.grid(row=2, column=3)
btn = Button(root, text='Rayleigh', command=lambda: put_into('result-up', ops.add_rayleigh_noise(manager.get_image('original-up'), gaussScale.get(), parameter.get())))
btn.grid(row=2, column=4)
btn = Button(root, text='Sal y Pimienta', command=lambda: put_into('result-up', ops.add_salt_pepper_noise(manager.get_image('original-up'), gaussScale.get())))
btn.grid(row=2, column=5)
filterScale = Scale(root, from_=2, to=15, orient=HORIZONTAL)
filterScale.set(5)
filterScale.grid(row=3, column=0)
btn = Button(root, text='Filtro media', command=lambda: put_into('result-up', ops.apply_mean_filter(manager.get_image('original-up'), filterScale.get())))
btn.grid(row=3, column=1)
btn = Button(root, text='Filtro mediana', command=lambda: put_into('result-up', ops.apply_median_filter(manager.get_image('original-up'), filterScale.get())))
btn.grid(row=3, column=2)
btn = Button(root, text='Filtro mediana ponderado', command=lambda: put_into('result-up', ops.apply_weighted_median_filter(manager.get_image('original-up'))))
btn.grid(row=3, column=3)
btn = Button(root, text='Filtro Gauss', command=lambda: put_into('result-up', ops.apply_gauss_filter(manager.get_image('original-up'))))
btn.grid(row=3, column=4)
btn = Button(root, text='Filtro pasaalto', command=lambda: put_into('result-up', ops.apply_pasaalto_filter(manager.get_image('original-up'))))
btn.grid(row=3, column=5)
btn = Button(root, text='Prewitt', command=lambda: put_into('result-up', bd.prewitt(manager.get_image('original-up'))))
btn.grid(row=4, column=0)
btn = Button(root, text='Sobel', command=lambda: put_into('result-up', bd.sobel(manager.get_image('original-up'))))
btn.grid(row=4, column=1)
btn = Button(root, text='Prewitt directional', command=lambda: put_into('result-up', bd.directional_prewitt(manager.get_image('original-up'))))
btn.grid(row=4, column=2)
btn = Button(root, text='Sobel directional', command=lambda: put_into('result-up', bd.directional_sobel(manager.get_image('original-up'))))
btn.grid(row=4, column=3)
btn = Button(root, text='Laplaciano', command=lambda: put_into('result-up', bd.laplace(manager.get_image('original-up'))))
btn.grid(row=5, column=0)
laplaceScale = Scale(root, from_=0, to=255, orient=HORIZONTAL)
laplaceScale.set(128)
laplaceScale.grid(row=5, column=1)
btn = Button(root, text='Laplaciano pendiente', command=lambda: put_into('result-up', bd.laplace_pendiente(manager.get_image('original-up'), laplaceScale.get())))
btn.grid(row=5, column=2)
sigmaScale = Scale(root, from_=1, to=11, orient=HORIZONTAL)
sigmaScale.set(1)
sigmaScale.grid(row=5, column=3)
btn = Button(root, text='Laplaciano gauss', command=lambda: put_into('result-up', bd.laplace_gauss(manager.get_image('original-up'), sigmaScale.get(), laplaceScale.get())))
btn.grid(row=5, column=4)
deltaTScale = Scale(root, from_=1, to=15, orient=HORIZONTAL)
deltaTScale.set(3)
deltaTScale.grid(row=6, column=0)
btn = Button(root, text='Umbralización global', command=lambda: put_into('result-up', umb.global_thresholding(manager.get_image('original-up'), deltaTScale.get())))
btn.grid(row=6, column=1)
btn = Button(root, text='Otsu', command=lambda: put_into('result-up', umb.otsu(manager.get_image('original-up'))))
btn.grid(row=6, column=2)
tScale = Scale(root, from_=1, to=5, orient=HORIZONTAL)
tScale.set(3)
tScale.grid(row=7, column=0)
sigScale = Scale(root, from_=20, to=100, resolution=5, orient=HORIZONTAL)
sigScale.set(20)
sigScale.grid(row=7, column=1)
btn = Button(root, text='Difusion Isotropica', command=lambda: put_into('result-up', diff.isotropic_diffusion(manager.get_image('original-up'), tScale.get())))
btn.grid(row=7, column=2)
btn = Button(root, text='Difusion Anisotropica', command=lambda: put_into('result-up', diff.anisotropic_diffusion(manager.get_image('original-up'), tScale.get(), sigScale.get(), funcSel.get())))
btn.grid(row=7, column=3)
funcSel = Scale(root, from_=1, to=2, orient=HORIZONTAL)
funcSel.set(1)
funcSel.grid(row=7, column=4)
btn = Label(root, text='TP3', font="Default 16 bold")
btn.grid(row=8, column=0)
btn = Button(root, text='Canny', command=lambda: put_into('result-up', bd.canny_detector(manager.get_image('original-up'))))
btn.grid(row=9, column=0)
btn = Button(root, text='SUSAN', command=lambda: put_into('result-up', bd.susan(manager.get_image('original-up'))))
btn.grid(row=9, column=1)
btn = Button(root, text='Hough', command=lambda: put_into('result-up', bd.hough_transform(manager.get_image('original-up'))))
btn.grid(row=9, column=2)
def active_contours():
    manager.restore_original()
    img = manager.get_image('original-up')
    img, _, _, _ = ac.active_contours_rect(img, motion_handler.get_rect())
    put_into('result-up', img)
btn = Button(root, text='Contornos activos', command=active_contours)
btn.grid(row=9, column=3)
btn = Button(root, text='Seleccionar region', command=start_choose_region)
btn.grid(row=9, column=4)


root.mainloop()
