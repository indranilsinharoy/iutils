# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          goptics.py
# Purpose:       Geometric optics utility functions
#
# Author:        Indranil Sinharoy
#
# Created:       06/24/2015
# Last Modified: 
# Copyright:     (c) Indranil Sinharoy 2015
# License:       MIT License
#-------------------------------------------------------------------------------
'''utility functions for geometric optics calculations 
'''
from __future__ import division, print_function
import numpy as _np
import math as _math


#%% Thick-lens formulae

class ThickLensInAir(object):
    '''Geometric computations of parameters for thick lens in air'''

    def __init__(self, r1, r2, t, n=1.5168):
        '''Thick lens in air specified by the radii of curvatures

        Parameters
        ---------- 
        r1 : float
            radius of curvature 1
        r2 : float
            radius of curvature 2
        d : float
            center thickness 
        n : float, optional 
            refractive index of the glass specified at the working wavelength.
            Default is 1.5168, which is the refractive index of N-BK7 at 0.5876 µm
        '''
        self.r1 = r1
        self.r2 = r2
        self.t = t
        self.n = n 

    @property
    def focal_length(self):
        '''focal length 
        '''
        r1, r2, t, n = self.r1, self.r2, self.t, self.n
        oneByf = (n-1)*(1/r1 - 1/r2 + (n-1)*t/(n*r1*r2)) 
        return 1/oneByf

    @classmethod
    def from_c(cls, c1, c2, t, n=1.5168):
        '''Thick lens in air specified by the curvatures 
        '''
        return cls(1/c1, 1/c2, t, n)

def gaussian_lens_formula(u=None, v=None, f=None, infinity=10e20):
    """return the third value of the Gaussian lens formula, given any two

    Parameters
    ----------
    u : float, optional
        object distance
    v : float, optional
        image distance
    f : float, optional
        focal length
    infinity : float
        numerical value to represent infinity (default=10e20)

    Returns
    -------
    value : float
        the third value given the other two of the Gaussian lens formula

    Examples
    --------
    >>> gaussian_lens_formula(u=1e20, f=10)
    10.0
    """
    if u:
        if v:
            f = (u*v)/(u+v)
            return f
        elif f:
            try:
                v = (u*f)/(u - f)
            except ZeroDivisionError:
                v = infinity
            return v
    else:
        try:
            u = (v*f)/(v - f)
        except ZeroDivisionError:
            u = infinity
        return u


#%% TEST FUNCTIONS

def _test_ThickLensInAir():
    '''test the class ThickLensInAir()
    '''
    lens = ThickLensInAir(r1=20.24, r2=-20.24, t=2.5) # edmund-optics Stock No. #63-537
    _nt.assert_almost_equal(20.0029519, lens.focal_length)
    lens2 = ThickLensInAir.from_c(c1=1/20.24, c2=-1/20.24, t=2.5)
    _nt.assert_almost_equal(20.0029519, lens2.focal_length)
    print("test_ThickLensInAir() class is successful") 


def _test_gaussian_lens_formula():
    """Test gaussian_lens_formula function"""
    v = gaussian_lens_formula(u=10e20, f=10)
    _nt.assert_equal(v, 10.0)
    v = gaussian_lens_formula(u=5000.0, f=100)
    _nt.assert_almost_equal(v, 102.04081632653062, decimal=5)
    u = gaussian_lens_formula(v=200, f=200)
    _nt.assert_equal(u, 10e20)
    f = gaussian_lens_formula(u=10e20, v=40)
    _nt.assert_almost_equal(f, 40, decimal=5)
    print("test_gaussian_lens_formula() is successful")

if __name__ == '__main__':
    import numpy.testing as _nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    # test functions
    _test_ThickLensInAir()
    _test_gaussian_lens_formula()