# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          imager.py
# Purpose:       camera and imaging system utility functions
#
# Author:        Indranil Sinharoy
#
# Created:       02/08/2014
# Last Modified: 02/08/2014
# Copyright:     (c) Indranil Sinharoy 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as np


#----------------------------------
# Digital sensor related functions
#----------------------------------

def pixelPitch(width, height, megapixels):
    """Return the pixel pitch for digital a sensor of given `width`, `height` and `megapixels`.
    
    It assumes 100% fill factor.
    
    Parameters
    ----------
    width : float
        width of the sensor in millimeters
    height : float
        height of the sensor in millimeters
    megapixels : float
        number of megapixels in the sensor
    
    Returns
    -------
    pixel_pitch : float
        pixel pitch in microns
    """
    return np.sqrt(width*height/megapixels)


def sensorNyqusitFrequency(pixel_pitch):
    """Return the nyquist frequency of the sensor
    
    Nyquist_frequency = Sampling_frequency /2 = 1/2*pixel_pitch
    
    Parameters
    ----------
    pixel_pitch : float
        pixel ptich of the sensor in microns
    
    Returns
    -------
    nyq_freq : float
        sensor Nyquist frequency in line-pair-per-millimeter (LPMM)
    """
    return 500.0/pixel_pitch
    
def lpph2lpmm(lpph, N, pixel_pitch):
    """Convert resolution specified in Line-Pair per Picture Height (LPPH) to Line-Pair per Milli Meters (LPMM)
    
    Parameters
    ----------
    lpph : float
        resolution in terms of Line-Pair per Picture Height (LPPH)
    N : integer
        Number of pixels in the picture along the dimension (height or width) of consideration
    pixel_pitch : float
        pixel pitch of the sensor in microns
    
    Returns
    -------
    lpmm : float
        resolution in terms of Line-Pair per Milli-Meters (LPMM)
    """
    return lpph*1000.0/(N*pixel_pitch)
    
def lpmm2lpph(lpmm, N, pixel_pitch):
    """Convert resolution specified in Line-Pair per Milli Meters (LPMM) to Line-Pair per Picture Height (LPPH)
    
    Parameters
    ----------
    lpmm : float
        resolution in terms of Line-Pair per Milli-Meters (LPMM)
    N : integer
        Number of pixels in the picture along the dimension (height or width) of consideration
    pixel_pitch : float
        pixel pitch of the sensor in microns
    
    Returns
    -------
    lpph : float
        resolution in terms of Line-Pair per Picture Height (LPPH)
    """
    return lpmm*N*pixel_pitch/1000.0
    

#---------------------------------------
# Imaging functions
#---------------------------------------
def geometricDepthOfField(focal_length, f_number, obj_dist, coc, grossExpr=False):
    """Returns the geometric depth of field value
    
    Parameters
    ----------
    focal_length : float 
        focal length in length units (usually in mm)
    f_number : float
        F/# of the optical system
    obj_dist : float
        distance of the object from the lens in focus in the same unit as `f`
    coc : float
        circle of confusion in the same length-unit as `f`
    grossExpr : boolean
        whether or not to use the expression from Gross' Handbook of Optical systems (see the notes). This is an approximation.
    
    Returns
    -------
    dof : float 
        length of the total depth of field zone in the same unit as `f`
    
    Notes
    -----
    The expression for the geometric depth of field is given as:
    
    DOF = 2*N*c*f^2*u^2/(f^4 - N^2*c^2*u^2
    
    The expression for the geometric depth of filed using Gross Expression, which is derived from the expression given in 30.8 of Handbook of Optical Systems, vol 3, Gross is as follows:
    
    DOF = 2*c*N*(u - f)^2/f^2
    """
    if grossExpr:
        dof = (2.0*coc*f_number*(obj_dist-focal_length)**2)/focal_length**2
    else:
        dof = (2.0*focal_length**2*f_number*coc*obj_dist**2)/(focal_length**4 - (f_number*coc*obj_dist)**2)
    return dof



if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.