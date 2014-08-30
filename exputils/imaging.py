# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          imaging.py
# Purpose:       quick utilities for imaging experiments
#
# Author:        Indranil Sinharoy
#
# Created:       08/30/2014
# Last Modified:
# Copyright:     (c) Indranil Sinharoy 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
"""imaging module containing a collection of utility functions useful
during imaging experiments.
"""
from __future__ import division, print_function
import math as _math
import numpy as _np
import warnings as _warnings
import collections as _co


# Unit conversion utilities
def mm2inch(val):
    return val/25.4

def inch2mm(val):
    return val*25.4

def ppi2microns(val):
    """convert LCD display resolution in PPI to microns"""
    return 1000*25.4/val


def get_ppi(device='surface_pro3'):
    """get display device resolution of known devices

    Parameters
    ----------
    device : string, optional
        name of the device. Options are -- 'surface_pro3', 'surface_pro2',
        'surface_pro1', 'nexus7',
    """
    ppi = None
    if device == 'surface_pro3':
        ppi = 216
    if device == 'surface_pro2':
        ppi = 208
    if device == 'surface_pro':
        ppi = 207.82
    if device == 'nexus7':
        ppi = 323
    return ppi

# first order imaging calculations
def mag(f, u=None, v=None):
    """returns the magnification value (without the sign) from focal-length
    and either one of the value of object or image distance

    Parameters
    ----------
    f : float
        focal length
    u : float, optional
        object distance
    v : float, optional
        image distance

    Returns
    -------
    m : float
        magnification (without the sign)
    """
    if v is not None:
        return (v - f)/f
    else:
        return f/(u - f)




# test functions

def _test_mm2inch():
    assert mm2inch(25.4) == 1.0

def _test_inch2mm():
    assert inch2mm(1) == 25.4

def _test_ppi2microns():
    ppi = get_ppi('surface_pro3')
    mic = ppi2microns(ppi)
    _nt.assert_almost_equal(117.592592593, mic)

def _test_mag():
    obj_dist = 3000
    foc = 200
    m = mag(foc, obj_dist)
    img_dist = _imgr.gaussian_lens_formula(u=obj_dist, f=foc)
    _nt.assert_almost_equal(img_dist/obj_dist, m, decimal=8)

    m = mag(foc, v=img_dist)
    obj_dist = _imgr.gaussian_lens_formula(v=img_dist, f=foc)
    _nt.assert_almost_equal(img_dist/obj_dist, m, decimal=8)

#
if __name__ == '__main__':
    import numpy.testing as _nt
    import iutils.opticsutils.imager as _imgr
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_mm2inch()
    _test_inch2mm()
    _test_ppi2microns()
    _test_mag()