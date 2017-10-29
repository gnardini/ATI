import numpy as np
from PIL import Image
from PIL import ImageTk

class ImageManagement:

    def __init__(self):
        self.panels = {
            'original-up': None,
            'original-down': None,
            'result-up': None,
            'result-down': None
        }
        self.images = {
            'original-up': None,
            'original-down': None,
            'result-up': None,
            'result-down': None
        }
        self.saved_original = None

    def get_panel(self, name):
        return self.panels[name]

    def set_panel(self, name, panel):
        self.panels[name] = panel

    def get_image(self, name):
        return self.images[name]

    def set_image(self, name, image):
        self.images[name] = image

    def save_original(self):
        self.saved_original = np.copy(self.get_image('original-up'))

    def restore_original(self):
        if self.saved_original is not None:
            self.put_into('original-up', np.copy(self.saved_original))

    def clear_cache(self):
        self.saved_original = None

    def to_tk_image(self, img):
        return ImageTk.PhotoImage(Image.fromarray(img))

    def put_into(self, key, img):
        self.set_image(key, img)
        img = self.to_tk_image(img)
        panel = self.get_panel(key)
        panel.configure(image=img)
        panel.image = img
