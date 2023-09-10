# Cartoonify

# Importing libraries

import sys
import os
import cv2
import easygui
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Reading an image

ImagePath=easygui.fileopenbox()
img=cv2.imread(ImagePath)
if img is None:
    print("ERROR: Can't find any image. Choose appropriate file path.")
    sys.exit()

# Converting image in RGB format

img=cv2.cvtColor(img,,cv2.COLOR_BGR2RGB)

# Converting image from RGB to grayscale

img_grey=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


# Blurring image

# Median Blurring (1)

img_blur=cv2.medianBlur(img_grey,3)
plt.imshow(img_blur)
plt.axis("off")
plt.title("AFTER MEDIAN BLURRING")
plt.show()

# Gaussian Blurring (2)

#####

# Bilateral Blurring (3)

#####

# Creating edge mask

edges=cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,3)
plt.imshow(edges)
plt.axis("off")
plt.title("Edge Mask")
plt.show()

# Erdoding & Dilating

kernel=np.ones((1,1),np.uint8)
img_erode=cv2.erode(img_bb,kernel,iterations=3)
img_dilate=cv2.dilate(img_erode,kernel,iterations=3)
plt.imshow(img_dilate)
plt.axis("off")
plt.title("AFTER ERODING AND DILATING")
plt.show()

# Stylization of image

img_style=cv2.stylization(img,sigma_s=150,sigma_r=0.25)
plt.imshow(img_style)
plt.axis("off")
plt.title("AFTER STYLIZATION")
plt.show()