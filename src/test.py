import matplotlib.pyplot as plt
import numpy as np
from images_function_tools import MyImage
from tkinter import filedialog


if __name__ =="__main__":
    img = MyImage.open_image_as_rgb_matrices(filedialog.askopenfilename())
    #v = np.array([[1,2,3],[4,5,6],[7,8,9]])
    #img = MyImage(v,v,v,mode='L')
    MyImage.show_image(img.rotate_simd(True))
