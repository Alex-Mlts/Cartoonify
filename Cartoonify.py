# Cartoonify

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def read_file(filename):
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #plt.imshow(img)
    #plt.title("Chosen image / BGR to RGB")
    #plt.show()
    return img

filename = "cat.jpg" # !Here add the name / path of the image you want to edit!
img = read_file(filename)

original = np.copy(img)

b_filter = cv.bilateralFilter(img, d = 3, sigmaColor = 200, sigmaSpace = 200)

#plt.imshow(b_filter)
#plt.title("Blurred image")
#plt.show()

def edge_mask(img, line_thickness, blur_value):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gblur = cv.medianBlur(gray, blur_value)

    edges = cv.adaptiveThreshold(gblur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, line_thickness, blur_value)

    return edges

line_thickness, blur_value = 5, 5
edges = edge_mask(img, line_thickness, blur_value)

#plt.imshow(edges, cmap="binary")
#plt.title("Edges in image")
#plt.show()

def color_quant(img, k):
    data = np.float32(img).reshape((-1,3))
    criteria = (cv.TERM_CRITERIA_EPS+ cv.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)

    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result

img_quant = color_quant(img, k = 5)

#plt.imshow(img_quant)
#plt.title("Major colors in image")
#plt.show()

def cartoonify():
    result = cv.bitwise_and(b_filter, img_quant, mask = edges)
    
    plt.imshow(original)
    plt.title("Original image")
    plt.show()

    plt.imshow(result)
    plt.title("Cartoonified image")
    plt.show()

cartoonify()
