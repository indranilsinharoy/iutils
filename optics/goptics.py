# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          goptics.py
# Purpose:       Geometric optics utility functions
#
# Author:        Indranil Sinharoy
#
# Created:       06/24/2015
# Last Modified: 12/04/2015
# Copyright:     (c) Indranil Sinharoy 2015
# License:       MIT License
#-------------------------------------------------------------------------------
'''utility functions for geometric optics calculations 
'''
from __future__ import division, print_function
import collections as _co
import numpy as _np
import math as _math


#%% Thick-lens formulae

class ThickLensInAir(object):
    '''Geometric computations of parameters for thick lens in air'''

    def __init__(self, r1, r2, t, n=1.5168):
        '''Thick lens in air specified by the radii of curvatures

        Parameters
        ---------- 
        r1 : float or np.inf
            radius of curvature of first surface
        r2 : float or np.inf
            radius of curvature of second surface
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
        r1 = 1/c1 if c1 else _np.inf
        r2 = 1/c2 if c2 else _np.inf     
        return cls(r1, r2, t, n)

def gaussian_lens_formula(u=None, v=None, f=None, infinity=10e20):
    """return the third value of the Gaussian lens formula, given any two

    Parameters
    ----------
    u : float, optional
        object distance from first principal plane. 
    v : float, optional
        image distance from rear principal plane 
    f : float, optional
        focal length
    infinity : float
        numerical value to represent infinity (default=10e20)

    Returns
    -------
    glfParams : namedtuple
        named tuple containing the Gaussian Lens Formula parameters

    Notes
    ----- 
    Both object and image distances are considered positive.   

    Examples
    --------
    >>> gaussian_lens_formula(u=30, v=None, f=10)
    glfParams(u=30, v=15.0, f=10)
    >>> gaussian_lens_formula(u=30, v=15)
    glfParams(u=30, v=15, f=10.0)
    >>> gaussian_lens_formula(u=1e20, f=10)
    glfParams(u=1e+20, v=10.0, f=10)
    """
    glfParams = _co.namedtuple('glfParams', ['u', 'v', 'f'])
    def unknown_distance(knownDistance, f):
        try: 
            unknownDistance = (knownDistance * f)/(knownDistance - f)
        except ZeroDivisionError:
            unknownDistance = infinity 
        return unknownDistance

    def unknown_f(u, v):
        return (u*v)/(u+v)

    if sum(i is None for i in [u, v, f]) > 1:
        raise ValueError('At most only one parameter can be None')

    if f is None:
        if not u or not v:
            raise ValueError('f cannot be determined from input')
        else:
            f = unknown_f(u, v)
    else:
        if u is None:
            u = unknown_distance(v, f)
        else:
            v = unknown_distance(u, f)
    return glfParams(u, v, f)


class TwoLensSystem(object):
    '''Combined lens sytem formed by combination of two lenses with focal 
       lengths `f1` and `f2` and thickness between them equal to `s` 

    Notes
    ----- 
    1. Combined focal length ``f: 1/f = 1/f1 + 1/f2 - s/(f1*f2)``
    2. BFL : ``f2*(s - f1)/(s - (f1 + f2))``. This is equivalent to ``f*(f1 - s)/f1``

    References
    ----------
    1. Lens Design Fundamentals, Rudolf Kingslake, R. Barry Johnson (Section 3.4.8)
    2. Telephoto lenses, Chapter 7, Lens design, Milton Laikin

    Examples
    -------- 
    lens = go.TwoLensSystem(75, -24, 60)
    >>>lens.f 
    200.0
    >>>lens.bfl 
    40.0
    >>>lens.s 
    100.0
    
    See Also
    -------- 
    The function `get_telephoto_pair()`  
    '''

    def __init__(self, f1, f2, s=0):
        """combined lens pair of focal lengths `f1` and `f2` with separation `s` 

        Parameters
        ---------- 
        f1 : real
            focal length for first lens 
        f2 : real 
            focal length of second lens 
        s : real, optional 
            space between the two lenses in the same lens units as the focal lengths.
            default=0
        """
        if f1:
            self._f1 = f1
        else:
            raise ValueError
        if f2:
            self._f2 = f2 
        else:
            raise ValueError  
        self._s = s

    @property
    def f1(self):
        return self._f1

    @property
    def f2(self):
        return self._f2
    
    @property
    def s(self):
        '''separation'''
        return self._s

    @f1.setter
    def f1(self, val):
        if val:
            self._f1 = val
        else:
            raise ValueError 

    @f2.setter
    def f2(self, val):
        if val:
            self._f2 = val
        else:
            raise ValueError 

    @s.setter
    def s(self, val):
        '''separation'''
        self._s = val  
         
    @property
    def f(self):
        '''focal length'''
        return  self._f1*self._f2/(self._f1 + self._f2 - self._s)

    @property
    def bfl(self):
        '''back focal length. i.e. distance from the final surface of the second 
        lens to the focal point of the combined lenses'''
        return self.f*(self._f1 - self._s)/self._f1 
        
    @property
    def t(self):
        '''distance from the first lens to the focal point
        '''
        return self.f + self._s - self._s*self.f/self._f1
    

def get_telephoto_pair(f, t, s=None):
    """returns the pair of positive and negative focal length lens pairs 
    that make up the telephoto lens.

    Parameters
    ---------- 
    f : real 
        focal length of the telephoto lens (i.e. focal length of the combined lens)
    t : real 
        total track length of the telephoto lens (i.e. distance from the first lens
        to the image plane at infinite focus)
    s : real, optional 
        separation between the pair of lenses. If `None`, `s` is set equal to 
        `s/2` for which the |f2| is maximum  

    Returns
    ------- 
    f1 : real 
        focal length of the positive lens (towards the object)
    f2 : real 
        focal length of the negetive lens (towards the image plane)

    Notes
    ----- 
    1. The telephoto ratio is equal to t/f
    2. bfl = t - s
    3. |f2| is maximum at s=t/2 

     |------------------->
              f      
          ^       ^
          |       |      |<--image plane
          |       |      |
     |----|-------|------|
     H'   |   s   |  bfl |
          |       |      |
          v       v       

       f1 (+)   f2 (-)     
   
          |-------------->
                 t

    References
    ----------
    1. Telephoto lenses, Chapter 7, Lens design, Milton Laikin

    Examples
    -------- 
    >>>go.get_telephoto_pair(f=200, t=100, s=60)
    (75.0, -24.0)          
    """
    if s is None:
        s = t/2.0
    f1 = s*f/(f - t + s)
    f2 = s*(s - t)/(f - t)
    return f1, f2

    


#%% TEST FUNCTIONS

def _test_ThickLensInAir():
    '''test the class ThickLensInAir()
    '''
    lens = ThickLensInAir(r1=20.24, r2=-20.24, t=2.5) # edmund-optics Stock No. #63-537
    _nt.assert_almost_equal(20.0029519, lens.focal_length)
    lens2 = ThickLensInAir.from_c(c1=1/20.24, c2=-1/20.24, t=2.5)
    _nt.assert_almost_equal(20.0029519, lens2.focal_length)
    print("test_ThickLensInAir() Class successful.\n") 


def _test_gaussian_lens_formula():
    """Test gaussian_lens_formula function"""
    v = gaussian_lens_formula(u=10e20, f=10).v
    _nt.assert_equal(v, 10.0)
    v = gaussian_lens_formula(u=5000.0, f=100).v
    _nt.assert_almost_equal(v, 102.04081632653062, decimal=5)
    u = gaussian_lens_formula(v=200, f=200).u
    _nt.assert_equal(u, 10e20)
    f = gaussian_lens_formula(u=10e20, v=40).f
    _nt.assert_almost_equal(f, 40, decimal=5)
    print("test_gaussian_lens_formula() successful.\n")
    
def _test_TwoLensSystem():
    '''test the class TwoLensSystem()
    '''
    lens = TwoLensSystem(f1=350.0, f2=350.0, s=20)
    _nt.assert_almost_equal(180.1470588235, lens.f)
    _nt.assert_almost_equal(169.8529411765, lens.bfl)
    print("test_TwoLensSystem() successful.\n")

if __name__ == '__main__':
    import numpy.testing as _nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    # test functions
    _test_ThickLensInAir()
    _test_gaussian_lens_formula()
    _test_TwoLensSystem()