# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          imager.py
# Purpose:       camera and imaging system utility functions
#
# Author:        Indranil Sinharoy
#
# Created:       02/08/2014
# Last Modified: 07/04/2014
# Copyright:     (c) Indranil Sinharoy 2014
# Licence:       MIT License
#-------------------------------------------------------------------------------
from __future__ import division, print_function
import numpy as _np
import math as _math

#----------------------------------
# Digital sensor related functions
#----------------------------------

def pixel_pitch(width, height, megapixels):
    """Return the pixel pitch for digital a sensor of given ``width``,
    ``height`` and ``megapixels``

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
    pixelPitch : float
        pixel pitch in microns
    """
    return _np.sqrt(width*height/megapixels)

def sensor_nyqusit_frequency(pixelPitch):
    """Return the nyquist frequency of the sensor

    Parameters
    ----------
    pixelPitch : float
        pixel ptich of the sensor in microns

    Returns
    -------
    nyqFreq : float
        sensor Nyquist frequency in line-pair-per-millimeter (LPMM)

    Notes
    -----
    .. math::

        F_{nyquist} = \\frac{F_s}{2 } = \\frac{1}{2*pixelPitch}
    """
    return 500.0/pixelPitch

def lpph2lpmm(lpph, n, pixelPitch):
    """Convert resolution specified in Line-Pair per Picture Height (LPPH)
    to Line-Pair per Milli Meters (LPMM)

    Parameters
    ----------
    lpph : float
        resolution in terms of Line-Pair per Picture Height (LPPH)
    n : integer
        Number of pixels in the picture along the dimension
        (height or width) of consideration
    pixelPitch : float
        pixel pitch of the sensor in microns

    Returns
    -------
    lpmm : float
        resolution in terms of Line-Pair per Milli-Meters (LPMM)
    """
    return lpph*1000.0/(n*pixelPitch)

def lpmm2lpph(lpmm, n, pixelPitch):
    """Convert resolution specified in Line-Pair per Milli Meters (LPMM) to
    Line-Pair per Picture Height (LPPH)

    Parameters
    ----------
    lpmm : float
        resolution in terms of Line-Pair per Milli-Meters (LPMM)
    n : integer
        Number of pixels in the picture along the dimension
        (height or width) of consideration
    pixelPitch : float
        pixel pitch of the sensor in microns

    Returns
    -------
    lpph : float
        resolution in terms of Line-Pair per Picture Height (LPPH)
    """
    return lpmm*n*pixelPitch/1000.0


#---------------------------------------
# Imaging functions
#---------------------------------------
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

def geometric_depth_of_field(focalLength, fNumber, objDist, coc, grossExpr=False):
    """Returns the geometric depth of field value

    Parameters
    ----------
    focalLength : float
        focal length in length units (usually in mm)
    fNumber : float
        F/# of the optical system
    objDist : float
        distance of the object from the lens in focus in the same unit
        as ``f``
    coc : float
        circle of confusion in the same length-unit as ``f``
    grossExpr : boolean
        whether or not to use the expression from Gross' Handbook of
        Optical systems (see the notes). This is an approximation.

    Returns
    -------
    dof : float
        length of the total depth of field zone in the same unit as ``f``

    Notes
    -----
    The expression for the geometric depth of field is given as:

    .. math::

        DOF = \\frac{2 N c f^2 u^2}{f^4 - N^2 c^2 u^2}

    The expression for the geometric depth of filed using Gross
    Expression, which is derived from the expression given in
    30.8 of Handbook of Optical Systems, vol 3, Gross is as follows:

    .. math::

        DOF = \\frac{2 c N (u - f)^2}{f^2}
    """
    if grossExpr:
        dof = (2.0*coc*fNumber*(objDist-focalLength)**2)/focalLength**2
    else:
        dof = ((2.0*focalLength**2*fNumber*coc*objDist**2)/
               (focalLength**4 - (fNumber*coc*objDist)**2))
    return dof


#---------------------------------------
# simple Scheimpflug camera
#---------------------------------------
# This is a very simple rudimentary version for quick calculations ...
# will extended it as we go ... may even break it up into components like
# lens, sensor, etc.

class scheimpflug(object):
    def __init__(self, f, u=None, lensImgAng=0.0, atype='d'): # angle input/output is in degrees for now
        if atype not in ['d', 'r']:
            raise TypeError, 'Invalid angle'
        self._f = f
        self._u = u
        if u:
            self._v = gaussian_lens_formula(u=u, v=None, f=f)
        else:
            self._v = None
        if atype == 'd':
            self._lensImgAng_d = lensImgAng  # perhaps this will be a vector later
            self._lensImgAng_r = lensImgAng*_math.pi/180
        else:
            self._lensImgAng_r = lensImgAng
            self._lensImgAng_d = lensImgAng*180/_math.pi
        self._lensPosfAng_r = None
        self._lensPosfAng_d = None
        # modify this so that the lensPosfAng is calculated if lens lensImgAng is set

    @property
    def f(self):
        return self._f

#    @f.setter   # when this is set, what should change? u or v?
#    def f(self, value):
#        self._f = value

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        self._u = value
        self._v = gaussian_lens_formula(u=value, v=None, f=self._f)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = value
        self._u = gaussian_lens_formula(u=None, v=value, f=self._f)

    @property
    def lensImgAng_d(self):
        return self._lensImgAng_d

    @lensImgAng_d.setter
    def lensImgAng_d(self, value):
        self._lensImgAng_d = value
        self._lensImgAng_r = value*_math.pi/180
        # calculate lensPosfAng_d and lensPosfAng_r
        tanLensImgAng = _math.tan(self._lensImgAng_r)
        tanLensPosfAng = (self._u/self._f - 1)*tanLensImgAng
        lensPosfAng = _math.atan(tanLensPosfAng)
        self._lensPosfAng_r = lensPosfAng
        self._lensPosfAng_d = lensPosfAng*180/_math.pi

    @property
    def lensImgAng_r(self):
        return self._lensImgAng_r

    @lensImgAng_r.setter
    def lensImgAng_r(self, value):
        self.lensImgAng_d = value*180/_math.pi

    @property
    def lensPosfAng_d(self):
        return self._lensPosfAng_d

    @lensPosfAng_d.setter
    def lensPosfAng_d(self, value):
        self._lensPosfAng_d = value
        self._lensPosfAng_r = value*_math.pi/180
        # calculate lensImgAng_d and lensImgAng_r
        tanLensPosfAng = _math.tan(self._lensPosfAng_r)
        tanLensImgAng = (self._f/(self._u - self._f))*tanLensPosfAng
        lensImgAng = _math.atan(tanLensImgAng)
        self._lensImgAng_r = lensImgAng
        self._lensImgAng_d = lensImgAng*180/_math.pi

    @property
    def lensPosfAng_r(self):
        return self._lensPosfAng_r

    @lensPosfAng_r.setter
    def lensPosfAng_r(self, value):
        self.lensPosfAng_d = value*180/_math.pi

# ---------------------------
#   TEST FUNCTIONS
# ---------------------------

def _test_gaussian_lens_formula():
    """Test gaussian_lens_formula function"""
    v = gaussian_lens_formula(u=10e20, f=10)
    nt.assert_equal(v, 10.0)
    v = gaussian_lens_formula(u=5000.0, f=100)
    nt.assert_almost_equal(v, 102.04081632653062, decimal=5)
    u = gaussian_lens_formula(v=200, f=200)
    nt.assert_equal(u, 10e20)
    f = gaussian_lens_formula(u=10e20, v=40)
    nt.assert_almost_equal(f, 40, decimal=5)
    print("test_gaussian_lens_formula() is successful")

if __name__ == '__main__':
    import numpy.testing as nt
    from numpy import set_printoptions
    set_printoptions(precision=4, linewidth=85)  # for visual output in manual tests.
    _test_gaussian_lens_formula()