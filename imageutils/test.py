# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 03:29:03 2014

@author: Indranil
"""
from __future__ import print_function, division
import numpy as np
import tifffile.tifffile as tf
import matplotlib.pyplot as plt

image = tf.imread('test.tif')

print(type(image))
print(image.shape)
print(image.dtype)
for i in range(3):
    print(np.max(image[:,:,i]))
    print(np.min(image[:,:,i]))


plt.imshow(image[:,:,1])
plt.show()
