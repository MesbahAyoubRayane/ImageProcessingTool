import matplotlib.pyplot as plt
import numpy as np
from images_function_tools import MyImage
from tkinter import filedialog
import json
import time

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
    K = 3
    img.show_image()
    cluters = img.kmean(K)
    images = []
    for i in range(len(cluters)):
        if len(cluters[i]) == 0:continue
        images.append(MyImage.new_from_pixels(cluters[i],img.mode,img.width,img.height))
    MyImage.show_images(images)
if __name__ =="__main__":
    img = MyImage.open_image_as_rgb_matrices(filedialog.askopenfilename())
    test_clustering(img)
    #test_histograme_based_operations(img)
    """img = MyImage.open_image_as_rgb_matrices(filedialog.askopenfilename())
    img = img.rescale(2,2)
    start = time.time()
    imrpvedimg = img.rotate_imp(30)
    end = time.time()
    improved = end - start
    print(f"improved {improved}")
    start = time.time()
    rotateimg = img.rotate(30)
    end = time.time()
    old = end - start
    print(f"old {old}")

    if old < improved:
        print(f"old is better than imrpved {improved - old}")
    else:
        print(f"improved is better than old {old - improved }")
    MyImage.show_images([imrpvedimg,rotateimg])"""
    #img.rescale(0.1,0.1).rescale(10,10).save_image("test.png")
    #img.rotate(30).show_image()
    #img.color_segmt(70).show_image()
    #test_clustering(img)
    #array = np.ones((5,5))
    #img.mean_filter(11).show_image()
    #test_geometric_operations(img)
    #img.rotate(30,"","ANTI_CLOCK_WISE").show_image()
    #test_histograme_based_operations(img)
    #test_create_histograme_functions(img)
    #img.gaussian_filter(11,3).histo_translation(20).show_image()
