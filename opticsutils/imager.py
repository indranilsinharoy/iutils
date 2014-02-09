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



if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.