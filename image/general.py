# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          general.py
# Purpose:       General image processing utilities
#                 1. handling I/O and display 
#
# Author:        Indranil Sinharoy
#
# Created:       29/08/2012
# Last Modified: 28/01/2014
# Copyright:     (c) Indranil Sinharoy 2012, 2013, 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import print_function, division
import os as _os
from PIL import Image as _Image
import numpy as _np
import iutils.plot.mplutils as _mpu

def cv2mpl(im):
    """Convert color image (numpy array) created using OpenCV's ``imread``
    to matplotlib's color image for displaying using ``imshow()``

    Color image in openCV is B,G,R. In Matplotlib it is R,G,B.

    Parameters
    ----------
    im : ndarray
        color image array created using OpenCV's ``imread`` function

    Returns
    -------
    image : ndarray
        color image as represented in Matplotlib image
    """
    nm = _np.zeros(im.shape, dtype=im.dtype)
    nm[:,:,0] = im[:,:,2]
    nm[:,:,1] = im[:,:,1]
    nm[:,:,2] = im[:,:,0]
    return nm

def get_imlist(filePath, itype='jpeg'):
    """Returns a list of filenames for all images of specified type in a
    directory

    Parameters
    ----------
    filePath : string
        full path name of the directory to be searched
    itype : string, optional
        type of images to be searched, for example -- 'jpeg', 'tiff', 'png',
        'dng', 'bmp' (without the dot(.))

    Returns
    -------
    imageFiles : list of strings
        list of image filenames with full path.
    """
    imlist = []
    opJoin = _os.path.join
    dirList = _os.listdir(filePath)
    if itype in ['jpeg', 'jpg']:
        extensions = ['.jpg', '.jpeg', '.jpe',]
    elif itype in ['tiff', 'tif']:
        extensions = ['.tiff', '.tif']
    else:
        extensions = [''.join(['.', itype.lower()]), ]
    for ext in extensions:
        imlist += [opJoin(filePath, f) for f in dirList if f.lower().endswith(ext)]
    return imlist

"""
#Alternative get_imlist using glob module
import glob
def alt_get_imlist(pat):
    return[os.path.join(pat,f) for f in glob.glob("*.jpg")]
"""

def imresize(image, size, rsfilter='ANTIALIAS'):
    """Resize an image array using PIL

    Parameters
    ----------
    image : ndarray
        input image to resize
    size : tuple
        the size of the output image (width, height)
    filter : PIL filter
        'NEAREST' for nearest neighbour, 'BILINEAR' for linear interpolation
        in a 2x2 environment, 'BICUBIC' for cubic spline interpolation in a
        4x4 environment, or 'ANTIALIAS' for a high-quality downsampling filter.

    Returns
    ------- 
    rimg : ndarray 
        resized image
    """
    pil_im = _Image.fromarray(_np.uint8(image))
    pilfilter = {'NEAREST':_Image.NEAREST, 'BILINEAR':_Image.BILINEAR,
                 'BICUBIC':_Image.BICUBIC, 'ANTIALIAS':_Image.ANTIALIAS}
    return _np.array(pil_im.resize(size, pilfilter[rsfilter]))


def histeq(image, nbr_bins=256):
    """Histogram equalization of a grayscale image.

    Parameters
    ----------
    im : ndarray (ndim=2)
        image array
    nbr_bins : int
        number of bins

    Returns
    -------
    image : ndarray
        histogram equalized image
    cdf :

    Notes
    -----
    This implementation is from "Programming Computer Vision using Python",
    by J.E. Solem
    """
    # get image histogram
    imhist, bins = _np.histogram(image.flatten(), nbr_bins, density=True) #returns normalized pdf
    cdf = imhist.cumsum()      # cumulative distribution function
    cdf = 255.0*cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = _np.interp(image.flatten(), bins[:-1], cdf)
    return im2.reshape(image.shape), cdf

def compute_average(imlist):
    """Compute the average of a list of images

    Parameters
    ----------
    imlist : list
        list of image files

    Notes
    -----
    This implementation is from "Programming Computer Vision using Python",
    by J.E. Solem
    """
    # Open first image and make into array of type float
    averageim = _np.array(_Image.open(imlist[0]), 'f')
    divisor = 1.0
    for imname in imlist[1:]:
        try:
            averageim += _np.array(_Image.open(imname))
            divisor +=1
        except:
            print(imname + '...skipped')
    averageim /= divisor
    # return average as uint8
    return _np.array(averageim,'uint8')


def imshow(image, fig=None, axes=None, subplot=None, interpol=None,
           xlabel=None, ylabel=None, figsize=None, cmap=None):
    """Rudimentary image display routine, for quick display of images without
    the axes

    Returns
    ------- 
    imPtHandle, fig, axes
    """
    return _mpu.imshow(image,fig, axes, subplot, interpol, xlabel, ylabel, figsize, cmap)