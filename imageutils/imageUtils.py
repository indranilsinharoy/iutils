#-------------------------------------------------------------------------------
# Name:          imageUtils.py
# Purpose:       Image processing (especially handling I/O and display) utilities
#
# Author:        Indranil Sinharoy
#
# Created:       29/08/2012
# Last Modified: 28/01/2014
# Copyright:     (c) Indranil Sinharoy 2012, 2013, 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def cv2mpl(im):
    """Convert 3-D color image numpy array created using OpenCV's imread to
        matplotlib's 3-D color image for displaying
        color image in openCV is B,G,R, in matplotlib it is
       R,G,B. Assuming color image
    """
    nm = np.zeros(im.shape,dtype=im.dtype)
    nm[:,:,0] = im[:,:,2]
    nm[:,:,1] = im[:,:,1]
    nm[:,:,2] = im[:,:,0]
    return nm

def get_imlist(file_path, image_type='JPEG'):
    """Returns a list of filenames for all images of specified type in a directory

    Parameters
    ---------- 
    file_path : string
        full path name of the directory to be searched
    image_type : string ('JPEG', 'TIFF', 'PNG')
        type of images to be searched, options are -- JPEG, TIFF, PNG
    
    Returns
    -------
    image_files : list of strings
        list of image filenames with full path.    
    """
    ext = '.jpg'
    if image_type == 'TIFF':
        ext = '.tiff'
    elif image_type == 'PNG':
        ext = '.png'
    return [os.path.join(file_path,f) for f in os.listdir(file_path) if f.endswith(ext)]

"""
#Alternative get_imlist using glob module
import glob    
def alt_get_imlist(pat):
    return[os.path.join(pat,f) for f in glob.glob("*.jpg")]
"""
   
def imresize(im, sz):
    """Resize an image array using PIL"""
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize(sz))


def histeq(im, nbr_bins=256):
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
    
    Note
    ----    
    This is the implementation from Programming Computer Vision using Python, by J.E. Solem
    """
    #get image histogram
    imhist, bins = np.histogram(im.flatten(),nbr_bins,density=True) #returns normalized pdf
    cdf = imhist.cumsum()      # cumulative distribution function
    cdf = 255.0*cdf / cdf[-1]  # normalize
    
    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    
    return im2.reshape(im.shape), cdf
    
def compute_average(imlist):
    """Compute the average of a list of images
       
    Parameters
    ---------
    imlist : list
        list of image files
    
    Note
    ----
    This is the implementation from Programming Computer Vision using Python, by J.E. Solem
    """
    #Open first image and make into array of type float
    averageim = np.array(Image.open(imlist[0]),'f')
    divisor = 1.0
    
    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
            divisor +=1
        except: 
            print imname + '...skipped'
    averageim /= divisor
    
    #return average as uint8
    return np.array(averageim,'uint8')
    
def myimshow(image, bGray=False, fig=None, axes=None, subplot=None, xlabel=None, ylabel=None):
    """My own redimentary image display routine"""
    if (subplot == None):
        subplot = int(111)        
    if(fig==None): #Open a figure window
        fig = plt.figure()
        axes = fig.add_subplot(subplot)
    elif(axes==None):
        axes = fig.add_subplot(subplot)
    if(bGray==True):
        plt.gray()
        print '\ngray'
    #plot the image
    imPtHandle = plt.imshow(image,cm.gray)
    #get the image height and width to set the axes limits    
    pix_height = image.shape[0]
    pix_width = image.shape[1]
    #Set the xlim and ylim to constrain the plot
    axes.set_xlim(0,pix_width-1)
    axes.set_ylim(pix_height-1,0)
    #Set the xlabel and ylable if provided
    if(xlabel != None):
        axes.set_xlabel(xlabel)
    if(ylabel != None):
        axes.set_ylabel(ylabel)
    #Make the ticks to empty list
    axes.get_xaxis().set_ticks([])
    axes.get_yaxis().set_ticks([])
    return imPtHandle, fig, axes
                    
    
    
    
    
    
    