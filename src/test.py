import matplotlib.pyplot as plt
import numpy as np
from images_function_tools import MyImage
from tkinter import filedialog


def test_create_histograme_functions(img:MyImage):
    img = img.gray_scale()
    print(img.create_histograme())
    print(img.create_cumulated_histograme())
    print(img.create_normilized_histograme())
    y = img.create_cumulative_normilized_histograme()
    plt.plot(range(256),y)
    plt.show()

def test_histograme_based_operations(img:MyImage):
    MyImage.show_images([img.gray_scale().remove_outliers("mean",3).expansion_dynamique(),img,img.heat_map()])
    return
    img.inverse().show_image()
    img.gray_scale().inverse().show_image()

if __name__ =="__main__":
    img = MyImage.open_image_as_rgb_matrices(filedialog.askopenfilename())
    test_histograme_based_operations(img)
    #test_create_histograme_functions(img)
    #img.gaussian_filter(11,3).histo_translation(20).show_image()
