import matplotlib.pyplot as plt
import numpy as np
from images_function_tools import MyImage
from tkinter import filedialog


def main():
    path =  filedialog.askopenfilename()
    img:MyImage = MyImage.open_image_as_rgb_matrices(path)
    MyImage.show_image(img.mean_filter_center(3))

if __name__ == "__main__":
    main()