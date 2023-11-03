import matplotlib.pyplot as plt
import numpy as np
from images_function_tools import MyImage
from tkinter import filedialog


if __name__ =="__main__":
    img = MyImage.open_image_as_rgb_matrices(filedialog.askopenfilename())
    img.gaussian_filter(11,3).histo_shift(20).show_image()