import matplotlib.pyplot as plt
import numpy as np
from images_function_tools import MyImage
from tkinter import filedialog
import json
import time

from gui import StackFrame

def test_create_histograme_functions(img:MyImage):
    img = img.gray_scale()
    print(img.histograme())
    print(img.cumulated_histograme())
    print(img.normilized_histograme())
    y = img.cumulative_normilized_histo()
    plt.plot(range(256),y)
    plt.show()

def test_histograme_based_operations(img:MyImage):
    img = img.gray_scale().remove_outliers("mean",3).histo_translation(50)
    x = img.histo_expansion_dynamique()
    y = img.histo_equalisation()
    MyImage.show_images([x,y,img])
    return 
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
    img.rescale(0.5,0.5).rescale(2,2).show_image()
    return
    img.gray_scale().rescale(2,2).show_image()
    return
    img.gray_scale().paste(1000,1000,(100,100)).translation((100,200)).show_image()
    img.gray_scale().paste(1000,1000,(100,100)).translation((300,200)).show_image()

def test_clustering(img:MyImage):
    K = 2
    img = img.gray_scale().histo_equalisation()
    img.show_image()
    cluters = img.kmean(K)
    images = []
    for i in range(len(cluters)):
        if len(cluters[i]) == 0:continue
        images.append(MyImage.new_from_pixels(cluters[i],img.mode,img.width,img.height))
    MyImage.show_images(images)


"""def test_imp_gaussian(img:MyImage):
    k = 11
    img.show_image()
    start = time.time()
    imp = img.gaussian_filter_imp(k,20)
    end = time.time()
    print(f"improved gaussian filter = {end - start}")
    start = time.time()
    old = img.gaussian_filter(k,20)
    end = time.time()
    print(f"old gaussian filter = {end - start}")
    MyImage.show_images([old,imp])"""


"""def test_imp_mean(img:MyImage):
    s = 51
    img.show_image()
    start = time.time()
    imp = img.mean_filter_imp(s)
    end = time.time()
    print(f"improved mean filter = {end - start}")
    start = time.time()
    old = img.mean_filter(s)
    end = time.time()
    print(f"old mean filter = {end - start}")
    MyImage.show_images([old,imp])"""

def test_median_filter(img:MyImage):
    s = 11
    MyImage.show_images([img.median_filter(s),img])

def test_binary_taging(img:MyImage):
    k = 5
    imgs =[list(img.pixels())]
    imgs.extend(list(filter(lambda x: len(x) > 0, img.kmean(k))))
    imgs = [MyImage.new_from_pixels(p,img.mode,img.width,img.height) for p in imgs]
    MyImage.show_images(imgs)
    i = None
    while True:
        i = input("select the image to tag")
        try:
            i = int(i)
            break
        except Exception as _:
            continue
    img_to_tag = imgs[i]
    img_to_tag.binary_tagging().show_image()


if __name__ =="__main__":
    img = MyImage.open_image_as_rgb_matrices(filedialog.askopenfilename()).median_filter(5)
    img.edge_detection_robert(20).show_image()