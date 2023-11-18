import matplotlib.pyplot as plt
import numpy as np
from images_function_tools import MyImage
from tkinter import filedialog


def test_create_histograme_functions(img:MyImage):
    img = img.gray_scale()
    print(img.histograme())
    print(img.cumulated_histograme())
    print(img.normilized_histograme())
    y = img.cumulative_normilized_histo()
    plt.plot(range(256),y)
    plt.show()

def test_histograme_based_operations(img:MyImage):
    model = MyImage.open_image_as_rgb_matrices(filedialog.askopenfilename())
    model = model.gray_scale()
    img = img.gray_scale()
    MyImage.show_images([img,img.histo_matching(model.cumulative_normilized_histo()),model])
    return 
    MyImage.show_images([img.histo_equalisation(),img])
    return
    x = img.gray_scale().remove_outliers("mean",3).histo_expansion_dynamique()
    x.show_histogram()
    MyImage.show_images([x,img,img.heat_map()])
    return
    img.histo_inverse().show_image()
    img.gray_scale().histo_inverse().show_image()

def test_geometric_operations(img:MyImage):
    img.gray_scale().paste(1000,1000,(100,100)).translation((100,200)).show_image()

if __name__ =="__main__":
    img = MyImage.open_image_as_rgb_matrices(filedialog.askopenfilename())
    #test_geometric_operations(img)
    img.rotate(60,"").show_image()
    #img.rotate(30,"","ANTI_CLOCK_WISE").show_image()
    #test_histograme_based_operations(img)
    #test_create_histograme_functions(img)
    #img.gaussian_filter(11,3).histo_translation(20).show_image()
